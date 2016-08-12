require 'torch'

local CharacterMap = torch.class('CharacterMap')

function CharacterMap:__init(index2word)
  self.i2w = index2word
  self.max_len, self.c2i, self.i2c = self.collect_char_stats(index2word)
end

function CharacterMap:getCharSeq(i, output)
  local word = self.i2w[i]
  local last_index = 1
  if not output then
    output = torch.IntTensor(self.max_len + 2)
    output:fill(1)
  end
  for c in string.gmatch(word, '.') do
    last_index = last_index + 1
    output[last_index] = self.c2i[c]
  end
  return output
end

function CharacterMap:getCharSeqs(indexes)
  local output = torch.IntTensor(indexes:size(1), self.max_len + 2)
  output:fill(1)
  for i = 1, indexes:size(1) do
    self:getCharSeq(indexes[i], output[i])
  end
  return output
end

function CharacterMap.collect_char_stats(index2word)
  local max_len = 0
  local last_index = 1
  local char2index = {['##']=1}
  local index2char = {[1]='##'}
  for _, word in pairs(index2word) do
    if #word > max_len then max_len = #word end
    for c in string.gmatch(word, '.') do
      if not char2index[c] then
        last_index = last_index + 1
        char2index[c] = last_index
        index2char[last_index] = c
      end
    end
  end
  return max_len, char2index, index2char
end

return char_util
