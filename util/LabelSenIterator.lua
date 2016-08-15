require 'torch'

local LabelSenIterator = torch.class('LabelSenIterator')

function LabelSenIterator:__init(config)
    config = config or {}
    local args
    args, self.train_file, self.valid_file, self.test_file,
          self.batch_size, self.seq_length,
          self.val_batch_size, self.test_batch_size,
          self.sos_id, self.eos_id, self.padding_id, self.n_skip
        = xlua.unpack(
        {config},
        'LabelSenIterator',
        'Load data files in torch binary format. Data will be padded to fit mini batches',
        {arg='train_t7', type='string', default='data/commondefs/train_l.t7',
         help='training data in torch binary '},
        {arg='valid_t7', type='string', default='data/commondefs/valid_l.t7',
         help='validating data in torch binary'},
        {arg='test_t7', type='string', default='data/commondefs/test_l.t7',
         help='testing data in torch binary'},
        {arg='batch_size', type='number', default=8,
         help='number of sequences to run for each mini batch'},
        {arg='seq_length', type='number', default=10,
         help='number of characters for each sequence'},
        {arg='val_batch_size', type='number', default=0,
         help='number of sequences to run for each mini batch (validation)'},
        {arg='test_batch_size', type='number', default=0,
         help='number of sequences to run for each mini batch (test)'},
        {arg='sos_id', type='number', default=2,
          help='Start of sentence Id.'},
        {arg='eos_id', type='number', default=1,
            help='End of sentence Id.'},
        {arg='padding_id', type='number', default=nil,
            help='Padding at the end of sentence.'},
        {arg='n_skip', type='number', default=0,
            help='Skip the first n tokens of each sentence'}
    )
    self.sen_dep = false -- This option does not make sense
    if not self.padding_id then self.padding_id = self.eos_id end
    -- setting up batch size
    local b = self.batch_size
    if self.val_batch_size < 1 then self.val_batch_size = b end
    if self.test_batch_size < 1 then self.test_batch_size = b end
    local bv, bt = self.val_batch_size, self.test_batch_size
    self.split_batch_size = {train=b, val=bv, test=bt}
    self.split_next_idxs = {}
    self.split_data = {}
    self.split_sen_map = {}
    self.split_sen_count = {}
    self.split_batch_idxs = {}
    self.split_batch_data ={}
    self.split_batch_start = {}
    self.split_batch_mask = {}
    self.split_batch_labels = {}
    self:initData(self.train_file, 'train')
    if not self.sen_dep then self:initBatch('train') end
    self:initData(self.valid_file, 'val')
    if not self.sen_dep then self:initBatch('val') end
    self:initData(self.test_file, 'test')
    if not self.sen_dep then self:initBatch('test') end
    collectgarbage()
end

function LabelSenIterator:initData(file_path, split)
  local data = torch.load(file_path)
  -- find out number of sentences
  local count, map = data.label:size(1), data.label
  local b = self.split_batch_size[split]
  -- create a random permutation of sentences' index
  local num_batch = math.ceil(count / b)
  local batch = torch.randperm(num_batch * b)
  batch:mod(count):add(1) -- a quick way to wrap around
  -- reshape the index into batch
  batch = batch:view(num_batch, -1)
  self.split_batch_idxs[split] = batch
  self.split_sen_count[split] = count
  self.split_sen_map[split] = map
  self.split_data[split] = data.data
  self.split_next_idxs[split] = 1
end

function LabelSenIterator:initBatch(split)
  -- get current batch of sentences index
  local idx = self.split_next_idxs[split]
  -- return nil if no more data
  if idx > self.split_batch_idxs[split]:size(1) then
    return nil, nil, nil
  end
  -- find the longest size and number of batchs (according to the seq len)
  local batch_idxs = self.split_batch_idxs[split][idx]
  local sentences, labels, max_len = self.collectSentences(
    batch_idxs,
    self.split_sen_map[split],
    self.split_data[split], self.n_skip)
  local b = self.split_batch_size[split]
  local nb = math.ceil((max_len + 1) / self.seq_length)
  local len = self.seq_length > 0 and nb * self.seq_length or max_len
  local data = torch.IntTensor(b, len + 1) -- +1 for targets
  local mask = torch.ShortTensor(b, len+1):fill(0)
  data:fill(self.padding_id)

  -- for each sentence, copy data into the tensor (left align)
  local num_tokens = 0
  for i = 1, #sentences do
    data:select(1, i)[1] = self.sos_id
    data:select(1, i):narrow(
      1, 2, sentences[i]:size(1)):copy(sentences[i])
    mask:select(1, i):narrow(
      1, 1, sentences[i]:size(1) + 1):fill(1)
    num_tokens = num_tokens + sentences[i]:size(1)
  end
  self.split_batch_start[split] = 1
  self.split_batch_data[split] = data
  self.split_batch_mask[split] = mask
  self.split_next_idxs[split] = idx + 1
  self.split_batch_labels[split] = labels
  return data, labels, mask
end

function LabelSenIterator:nextBatch(split)
  local batch_start = self.split_batch_start[split]
  local batch_data = self.split_batch_data[split]
  local batch_mask = self.split_batch_mask[split]
  local batch_labels = self.split_batch_labels[split]
  local new_sentence = false
  if batch_start == 1 then new_sentence = true end
  --  check if current batch end
  if batch_start >= batch_data:size(2) then
    batch_data, batch_labels, batch_mask = self:initBatch(split)
    batch_start = 1
    new_sentence = true
    -- return nothing if no more data (init batch)
    if not batch_data then return nil, nil, nil, nil end
  end
  -- return current batch portion of the batch data
  local batch_end = self.seq_length > 0 and
    batch_start + self.seq_length - 1 or
    batch_data:size(2) - 1
  local x = batch_data[{{}, {batch_start, batch_end}}]
  local y = batch_data[{{}, {batch_start + 1, batch_end + 1}}]
  local mask = batch_mask[{{}, {batch_start + 1, batch_end + 1}}]
  -- increment current batch
  batch_start = self.seq_length > 0 and batch_start + self.seq_length
                                    or batch_end + 1
  self.split_batch_start[split] = batch_start
  return {x=x, y=y, label=batch_labels, new=new_sentence, m=mask}
end

function LabelSenIterator:reset(split)
  local b = self.split_batch_size[split]
  local count = self.split_sen_count[split]
  -- create a random permutation of sentences' index
  local num_batch = math.ceil(count / b)
  local batch = torch.randperm(num_batch * b)
  batch:mod(count):add(1) -- a quick way to wrap around
  -- reshape the index into batch
  batch = batch:view(num_batch, -1)
  self.split_batch_idxs[split] = batch
  self.split_next_idxs[split] = 1
  self:initBatch(split)
end

--[[Static methods]]--

function LabelSenIterator.collectSentences(batch_idxs, map, data, n_skip)
  local sentences = {}
  local labels = torch.zeros(batch_idxs:size(1)):type(map:type())
  local max_len = 0
  for i = 1, batch_idxs:size(1) do
    local from = map[batch_idxs[i]][1] + n_skip
    local to = batch_idxs[i] + 1
    to = to > map:size(1) and data:size(1) or map[to][1] - 1
    sentences[i] = data[{{from, to}}]
    labels[i] = map[batch_idxs[i]][2]
    local len = sentences[i]:size(1)
    if max_len < len then max_len = len end
  end
  return sentences, labels, max_len
end
