require 'torch'

local LMHelper = torch.class('LMHelper')

function LMHelper:__init(config)
  config = config or {}
  local args
  args, self.cuda, self.batch_size, self.rho, self.char_map,
        self.mode, self.ri_char_cnn, self.ri_hypernym,
        self.lm, self.data_dir, self.data_type,
        self.sos_id, self.eos_id, self.n_skip
      = xlua.unpack(
      {config},
      'LMHelper',
      'A collection of utility function to train network',
      {arg='cuda', type='boolean', default=true,
       help='Enable CUDA'},
      {arg='batchSize', type='number', default=32,
       help='Batch size of each mini-batche'},
      {arg='rho', type='number', default=5,
       help='Number of steps for BPTT'},
      {arg='charMap', type='object', default=nil,
       help='Character map object (for Char CNN model)'},
      {arg='mode', type='string', default='sen'},
      {arg='RICharCNN', type='boolean', default=false},
      {arg='RIHypernym', type='boolean', default=false},
      {arg='lm', type='object', default=nil,
       help='Model to train'},
      {arg='dataDir', type='string', default='data/commondefs'},
      {arg='dataType', type='string', default='sentence'},
      {arg='sosId', type='number', default=2},
      {arg='eosId', type='number', default=1},
      {arg='nSkips', type='number', default=0}
  )
  self._rho = self.rho < 1 and 1 or self.rho
  self.inputs = torch.Tensor(self._rho, self.batch_size)
  self.targets = torch.Tensor(self._rho, self.batch_size)
  self.m  = torch.Tensor(self.batch_size, self._rho)
  self.labels = torch.Tensor(self.batch_size)
  if self.cuda then
    self.inputs, self.labels = self.inputs:cuda(), self.labels:cuda()
    self.targets, self.m = self.targets:cuda(), self.m:cuda()
  end
  if self.char_map then
    self.chars = torch.Tensor(self.batch_size, self.char_map.max_len + 2)
    if self.cuda then
      self.chars = self.chars:cuda()
    end
  end

  if self.data_type == 'sentence' then
    include('SenIterator.lua')
    self.loader = SenIterator{
      train_t7= path.join(self.data_dir, 'train.t7'),
      valid_t7= path.join(self.data_dir, 'valid.t7'),
      test_t7= path.join(self.data_dir, 'test.t7'),
      batch_size=self.batch_size,
      seq_length=self.rho,
      sos_id=self.sos_id,
      eos_id=self.eos_id,
      padding_id=self.eos_id
    }
  else
    include('LabelSenIterator.lua')
    self.loader = LabelSenIterator{
      train_t7= path.join(self.data_dir, 'train_l.t7'),
      valid_t7= path.join(self.data_dir, 'valid_l.t7'),
      test_t7= path.join(self.data_dir, 'test_l.t7'),
      batch_size=self.batch_size,
      seq_length=self.rho,
      sos_id=self.sos_id,
      eos_id=self.eos_id,
      padding_id=self.eos_id,
      n_skip=self.n_skip
    }
  end
end

function LMHelper:_check_size(batch)
  local b = batch.x:size(1)
  local rho = batch.x:size(2)
  self._rho = rho
  if b ~= self.inputs:size(2) or rho ~= self.inputs:size(1) then
    self.inputs = torch.Tensor(rho, b)
    self.targets = torch.Tensor(rho, b)
    self.m  = torch.Tensor(b, rho)
    self.labels = torch.Tensor(b)
    if self.char_map then
      self.chars = torch.Tensor(b, self.char_map.max_len + 2)
      if self.cuda then
        self.chars = self.chars:cuda()
      end
    end
    if self.cuda then
      self.inputs, self.labels = self.inputs:cuda(), self.labels:cuda()
      self.targets, self.m = self.targets:cuda(), self.m:cuda()
    end
  end
end

function LMHelper:predict(batch)
  if batch.new then
    lm:forget()
  end
  self:_check_size(batch)

  self.inputs:copy(batch.x:t())
  self.targets = batch.y and self.targets:copy(batch.y:t()) or nil
  self.m = batch.m and self.m:copy(batch.m) or self.m:fill(1)

  self.input_word_embs = self.lookup:forward(self.inputs)
  self._lm_input = nil
  if self.mode == 'sen' then self._lm_input = self.input_word_embs end
  if self.mode == 'ri' then
    self.labels:copy(batch.label)
    self.label_word_embs = self.lookup_ri:forward(self.labels)
    self._lm_input = {self.label_word_embs, self.input_word_embs}
  end
  local ri_extra
  if self.ri_char_cnn or self.ri_hypernym then
    ri_extra = {self.label_word_embs}
  end
  if self.ri_char_cnn then
    self.chars:copy(self.char_map:getCharSeqs(batch.label))
    table.insert(ri_extra, self.chars)
  end
  if self.ri_hypernym then
    self.hypernym_embs = self.hlookup:forward(self.labels)
    table.insert(ri_extra, self.hypernym_embs)
  end
  if ri_extra then
    self._lm_input = {ri_extra, self.input_word_embs}
  end
  local predictions = self.lm:forward(self._lm_input)
  return predictions
end

function LMHelper:dlm(grad_predictions)
  self.lm:backward(self._lm_input, grad_predictions)
end

function LMHelper:maskGradPred(grad_predictions)
  for i = 1, self._rho do
    local mask = self.m[{{}, {i}}]
    if mask:sum() ~= self.batch_size then
      mask = mask:expandAs(grad_predictions[i])
      grad_predictions[i]:cmul(mask)
    end
  end
end

function LMHelper:clearModelState()
  self.lm:forget()
  self.lm:clearState()
end

function LMHelper.fileExists(name)
  local f=io.open(name,"r")
  if f~=nil then io.close(f) return true else return false end
end

function LMHelper.SeqClassNNLCriterion()
  local target = nn.Sequential()
  target:add(nn.SplitTable(1)) -- matrix to table of vectors
  target:add(
    cuda and nn.Sequencer(nn.Convert()) -- convert data type if cuda
         or  nn.Identity()
  )
  local nll = nn.ClassNLLCriterion()
  local crit = nn.ModuleCriterion(
    nn.SequencerCriterion(nll),
    nn.Identity(), -- input (nothing)
    target -- target
  )
  return crit
end
