require 'torch'

local word_util = {}

word_util.words2IdxTensor = function(words, w2i)
  local tensor = torch.IntTensor(#words)
  for i = 1, #words do
    tensor[i] = w2i[words[i]]
  end
  return tensor
end

word_util.loadIndexer = function(config)
  config = config or {}
  local opt =
    xlua.unpack(
      {config},
      'indexer',
      'Load indexers',
      {arg='dataDir', type='number', default='data/ptb_xtiny1b'},
      {arg='i2wFile', type='number', default='index2word.t7'},
      {arg='w2iFile', type='number', default='word2index.t7'}
  )
  local i2w = torch.load(path.join(opt.dataDir, opt.i2wFile))
  local w2i = torch.load(path.join(opt.dataDir, opt.w2iFile))
  return i2w, w2i
end

word_util.table2words = function(i2w, ids, eos_id, delimiter)
  delimiter = delimiter and delimiter or ' '
  local sentences = {}
  for i = 1, #ids do
    local sentence = {}
    for j = 1, #ids[i] do
      if ids[i][j] == eos_id then break end
      sentence[j] = i2w[ids[i][j]]
    end
    sentences[i] = stringx.join(delimiter, sentence)
  end
  return sentences
end

word_util.tensor2words = function(i2w, tensor, eos_id, delimiter)
  delimiter = delimiter and delimiter or ' '
  local sentences = {}
  for i = 1, tensor:size(1) do
    local sentence = {}
    for j = 1, tensor:size(2) do
      if tensor[i][j] == eos_id then break end
      sentence[j] = i2w[tensor[i][j]]
    end
    sentences[i] = stringx.join(delimiter, sentence)
  end
  return sentences
end

word_util.ids2words = function(i2w, data, eos_id, delimiter)
  if torch.type(data) == 'table' then
    return word_util.table2words(i2w, data, eos_id, delimiter)
  else
    return word_util.tensor2words(i2w, data, eos_id, delimiter)
  end
end

return word_util
