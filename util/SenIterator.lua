require 'torch'

local SenIterator = torch.class('SenIterator')

function SenIterator:__init(config)
    config = config or {}
    local args
    args, self.train_file, self.valid_file, self.test_file,
          self.batch_size, self.seq_length,
          self.val_batch_size, self.test_batch_size,
          self.sos_id, self.eos_id, self.padding_id, self.sen_dep
        = xlua.unpack(
        {config},
        'SentenceBatchLoader',
        'Load data files in torch binary format. Data will be padded to fit mini batches',
        {arg='train_t7', type='string', default='data/commondefs/train.t7',
         help='training data in torch binary '},
        {arg='valid_t7', type='string', default='data/commondefs/valid.t7',
         help='validating data in torch binary'},
        {arg='test_t7', type='string', default='data/commondefs/test.t7',
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
        {arg='sen_dep', type='boolean', default=false,
            help='Sentence dependent batch.'}
    )
    if self.seq_length < 1 then error('seq_length needs to be more than 0') end
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
    self.split_batch_len = {}
    self.split_batch_perm = {}
    self:initData(self.train_file, 'train')
    if not self.sen_dep then self:initBatch('train') end
    self:initData(self.valid_file, 'val')
    if not self.sen_dep then self:initBatch('val') end
    self:initData(self.test_file, 'test')
    if not self.sen_dep then self:initBatch('test') end
    collectgarbage()
end

function SenIterator:initSenInDepData(data, split)
  -- find out number of sentences
  local count, map = self.countSentences(data)
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
  self.split_data[split] = data[{{}, 2}]
  self.split_next_idxs[split] = 1
end

function SenIterator:initSenDepData(data, split)
  data = data[{{}, 2}]
  local b = self.split_batch_size[split]
  local total_len = data:size(1)
  local batch_len = math.ceil(total_len / b)
  self.split_batch_len[split] = batch_len
  self.split_batch_start[split] = 1
  self.split_data[split] = data
  self.split_batch_perm[split] = torch.randperm(b)
  self.split_batch_mask[split] = torch.ShortTensor(b, self.seq_length):fill(1)
end

function SenIterator:initData(file_path, split)
  local data = torch.load(file_path)
  if self.sen_dep then
    self:initSenDepData(data, split)
  else
    self:initSenInDepData(data, split)
  end
end

function SenIterator:initBatch(split)
  -- get current batch of sentences index
  local idx = self.split_next_idxs[split]
  -- return nil if no more data
  if idx > self.split_batch_idxs[split]:size(1) then
    return nil, nil
  end
  -- find the longest size and number of batchs (according to the seq len)
  local batch_idxs = self.split_batch_idxs[split][idx]
  local sentences, max_len = self.collectSentences(
    batch_idxs,
    self.split_sen_map[split],
    self.split_data[split])
  local b = self.split_batch_size[split]
  local nb = math.ceil((max_len + 1) / self.seq_length)
  local len = nb * self.seq_length
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
  return data, mask
end

function SenIterator:nextSenInDepBatch(split)
  local batch_start = self.split_batch_start[split]
  local batch_data = self.split_batch_data[split]
  local batch_mask = self.split_batch_mask[split]
  local new_sentence = false
  if batch_start == 1 then new_sentence = true end
  --  check if current batch end
  if batch_start >= batch_data:size(2) then
    batch_data, batch_mask = self:initBatch(split)
    batch_start = 1
    new_sentence = true
    -- return nothing if no more data (init batch)
    if not batch_data then return nil, nil, nil, nil end
  end
  -- return current batch portion of the batch data
  local batch_end = batch_start + self.seq_length - 1
  local x = batch_data[{{}, {batch_start, batch_end}}]
  local y = batch_data[{{}, {batch_start + 1, batch_end + 1}}]
  local mask = batch_mask[{{}, {batch_start + 1, batch_end + 1}}]
  -- increment current batch
  batch_start = batch_start + self.seq_length
  self.split_batch_start[split] = batch_start
  return {x=x, y=y, new=new_sentence, m=mask}
end

function SenIterator:nextSenDepBatch(split)
  local offset = self.split_batch_start[split]
  local batch_len = self.split_batch_len[split]
  if offset > batch_len then
    return nil
  end
  local b = self.split_batch_size[split]
  local l = self.seq_length
  local data = self.split_data[split]
  local batch_data = data.new():resize(b, l + 1)
  for i = 1, b do
    local perm = self.split_batch_perm[split][i]
    local start_idx = (perm - 1) * batch_len + offset
    local end_idx = (start_idx + l)
    if end_idx > data:size(1) then
      local s_l = data:size(1) - start_idx + 1
      local s_e = l - s_l + 1
      batch_data[{i, {1, s_l}}] = data[{{start_idx, data:size(1)}}]
      batch_data[{i, {1+s_l, l + 1}}] = data[{{1, s_e}}]
    else
      batch_data[i] = data[{{start_idx, end_idx}}]
    end
  end
  self.split_batch_start[split] = offset + l
  return {
    x = batch_data[{{}, {1, l}}],
    y = batch_data[{{}, {2, l+1}}],
    new = false,
    m = self.split_batch_mask[split]
  }
end

function SenIterator:nextBatch(split)
  if self.sen_dep then
    return self:nextSenDepBatch(split)
  else
    return self:nextSenInDepBatch(split)
  end
end

function SenIterator:reset(split)
  local b = self.split_batch_size[split]
  if self.sen_dep then
    self.split_batch_start[split] = 1
    self.split_batch_perm[split] = torch.randperm(b)
  else
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
end

--[[Static methods]]--

function SenIterator.countSentences(data)
  local prev = 0
  local count = 0
  local start_idxs = {}
  for i = 1, data:size(1) do
    if prev ~= data[i][1] then
        prev = data[i][1]
        count = count + 1
        table.insert(start_idxs, i)
    end
  end
  return count, torch.IntTensor(start_idxs)
end

function SenIterator.collectSentences(batch_idxs, map, data)
  local sentences = {}
  local max_len = 0
  for i = 1, batch_idxs:size(1) do
    local from = map[batch_idxs[i]]
    local to = batch_idxs[i] + 1
    to = to > map:size(1) and data:size(1) or map[to] - 1
    sentences[i] = data[{{from, to}}]
    local len = sentences[i]:size(1)
    if max_len < len then max_len = len end
  end
  return sentences, max_len
end
