require 'pl'
require 'torch'
require 'dp'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Preprocess a definition file for training a langauge model.')
cmd:option('--dataDir', 'data/commondefs', 'Data directory')
cmd:option('--trainFile', 'train.txt', 'Training text file')
cmd:option('--validFile', 'valid.txt', 'Validation text file')
cmd:option('--testFile', 'test.txt', 'Testing text file')
cmd:option('--trainT7', 'train.t7',
           'Output training data file in torch binary file')
cmd:option('--validT7', 'valid.t7',
           'Output validating data file in torch binary file')
cmd:option('--testT7', 'test.t7',
           'Output testing data file in torch binary file')
cmd:option('--trainLabelT7', 'train_l.t7',
          'Output training data file in torch binary file')
cmd:option('--validLabelT7', 'valid_l.t7',
          'Output validating data file in torch binary file')
cmd:option('--testLabelT7', 'test_l.t7',
          'Output testing data file in torch binary file')
cmd:option('--word2IndexT7', 'word2index.t7',
           'Output map from word to index in torch binary file')
cmd:option('--index2WordT7', 'index2word.t7',
          'Output map from index to word in torch binary file')
cmd:option('--freqT7', 'word_freq.t7',
           'Output map from word to frequency in torch binary file')
cmd:option('--unkSym', '<unk>', 'Unknown symbol')
cmd:option('--startSym', '<s>', 'Start sentence symbol')
cmd:option('--endSym', '</s>', 'End sentence symbol')
cmd:option('--defSym', '<def>', 'Definition symbol')
cmd:option('--minCount', -1, 'Minimum frequency, lower words will be unknown (not work with limitVocab)')
cmd:option('--limitVocab', -1, 'Number of words in vocab (not work with minCount)')
cmd:text()
opt = cmd:parse(arg or {})
table.print(opt)

--[[First pass: vocab]]--
function get_seq_tokens(line)
  local parts = stringx.split(line, '\t')
  local word = parts[1]
  local definition = parts[4]
  local output = word .. ' ' .. opt.defSym .. ' ' .. stringx.strip(definition) .. ' ' .. opt.endSym
  return stringx.split(output, ' ')
end

function collect_vocab(text_file)
  print(text_file)
  local f = io.open(text_file)
  wcount = 0
  while true do
    local line = f:read()
    if not line then break end
    local words = get_seq_tokens(line)
    for _, v in pairs(words) do
      wcount = wcount + 1
      if not word2index[v] then
        last_index = last_index + 1
        word2index[v] = last_index
        index2word[last_index] = v
      end
    end
  end
  f:close()
  return wcount
end

--[[Sencond Pass: Frequency]]--
function count_words(text_file)
  local f = io.open(text_file)
  while true do
    local line = f:read()
    if not line then break end
    local words = get_seq_tokens(line)
    for _, v in pairs(words) do
      local idx = word2index[v]
      word_freq[idx] = word_freq[idx] + 1
    end
  end
  f:close()
end

function limit_vocab(limit)
  if limit > word_freq:size(1) then return end
  local sorted, idx = torch.sort(word_freq, true)
  for i = limit, word_freq:size(1) do
    local word = index2word[idx[i]]
    if word ~= opt.startSym and word ~= opt.endSym and word ~= opt.unkSym then
      index2word[idx[i]] = nil
      word2index[word] = nil
    end
  end
  local _index2word, _word2index = {}, {}
  local cur = 1
  for k, v in pairs(index2word) do
    _index2word[cur] = v
    _word2index[v] = cur
    cur = cur + 1
  end
  index2word = _index2word
  word2index = _word2index
end

function cut_vocab(min_count)
  local _index2word, _word2index = {}, {}
  local cur = 1
  for k, v in pairs(index2word) do
    if word_freq[k] >= min_count or k <=3 then
      _index2word[cur] = v
      _word2index[v] = cur
      cur = cur + 1
    end
  end
  index2word = _index2word
  word2index = _word2index
end

--[[Third Pass: Data]]--
function make_data(text_file, wcount)
  local sentence_start_at = 1
  local cur_pos = 1
  local data = torch.IntTensor(wcount, 2)
  local f = io.open(text_file)
  while true do
    local line = f:read()
    if not line then break end
    local words = get_seq_tokens(line)
    for _, v in pairs(words) do
      local idx = word2index[v]
      if not idx then idx = unk_idx end
      word_freq[idx] = word_freq[idx] + 1
      data[cur_pos][1] = sentence_start_at
      data[cur_pos][2] = idx
      cur_pos = cur_pos + 1
      if idx == eos_idx then
        sentence_start_at = cur_pos
      end
    end
  end
  f:close()
  return data
end

--[[Forth Pass: Label data]]
function get_data(line)
  local parts = stringx.split(line, '\t')
  local word = parts[1]
  local definition = stringx.strip(parts[4])
  return word, stringx.split(word .. ' ' .. opt.defSym .. ' ' .. definition .. ' ' .. opt.endSym)
end

function count_defs(text_file)
  local f = io.open(text_file)
  local num_defs = 0
  local num_words_in_defs = 0
  while true do
    local line = f:read()
    if not line then break end
    local label, def = get_data(line)
    num_defs = num_defs + 1
    num_words_in_defs = num_words_in_defs + #def
  end
  f:close()
  return num_defs, num_words_in_defs
end

function make_label_data(text_file, lcount, tcount)
  local sentence_start_at = 1
  local cur_pos = 1
  local line_num = 1
  local data = torch.IntTensor(tcount)
  local label = torch.IntTensor(lcount, 2)
  local f = io.open(text_file)
  while true do
    local line = f:read()
    if not line then break end
    local word, def = get_data(line)
    label[line_num][1] = sentence_start_at
    label[line_num][2] = word2index[word]
    for _, v in pairs(def) do
      local idx = word2index[v]
      if not idx then idx = unk_idx end
      data[cur_pos] = idx
      cur_pos = cur_pos + 1
      if idx == eos_idx then
        sentence_start_at = cur_pos
      end
    end
    line_num = line_num + 1
  end
  f:close()
  local output = {}
  output.label = label
  output.data = data
  return output
end

print('First pass: collecting vocab')
eos_idx = 1
sos_idx = 2
unk_idx = 3
def_idx = 4
word2index = {[opt.endSym]=eos_idx, [opt.startSym]=sos_idx,
              [opt.unkSym]=unk_idx, [opt.defSym]=def_idx}
index2word = {[eos_idx]=opt.endSym, [sos_idx]=opt.startSym,
              [unk_idx]=opt.unkSym, [def_idx]=opt.defSym}
last_index = 4
train_wcount = collect_vocab(path.join(opt.dataDir, opt.trainFile))
valid_wcount = collect_vocab(path.join(opt.dataDir, opt.validFile))
test_wcount = collect_vocab(path.join(opt.dataDir, opt.testFile))
print(string.format('- Vocab size: %d', #index2word))

print('Second Pass: word frequency')
word_freq = torch.IntTensor(#index2word)
word_freq:zero()
count_words(path.join(opt.dataDir, opt.trainFile))
count_words(path.join(opt.dataDir, opt.validFile))
count_words(path.join(opt.dataDir, opt.testFile))
print(string.format('- Total number of tokens: %d', word_freq:sum()))
if opt.minCount > 0 then
  print('- Cutting vocab...')
  cut_vocab(opt.minCount)
elseif opt.limitVocab > 0 then
  print('- Limiting vocab...')
  limit_vocab(opt.limitVocab)
end
print(string.format('- Vocab size: %d', #index2word))
print('- saving...')
torch.save(path.join(opt.dataDir, opt.index2WordT7), index2word)
torch.save(path.join(opt.dataDir, opt.word2IndexT7), word2index)
torch.save(path.join(opt.dataDir, opt.freqT7), word_freq)

print('Third Pass: data')
word_freq = torch.IntTensor(#index2word)
word_freq:zero()
print('- building testing data...')
local test_data = make_data(path.join(opt.dataDir, opt.testFile), test_wcount)
torch.save(path.join(opt.dataDir, opt.testT7), test_data)
test_data = nil
print('- building validating data...')
local valid_data = make_data(path.join(opt.dataDir, opt.validFile), valid_wcount)
torch.save(path.join(opt.dataDir, opt.validT7), valid_data)
valid_data = nil
print('- building training data...')
local train_data = make_data(path.join(opt.dataDir, opt.trainFile), train_wcount)
torch.save(path.join(opt.dataDir, opt.trainT7), train_data)
train_data = nil

print('Forth Pass: label data')
print('- counting definitions...')
train_defs, train_tokens = count_defs(path.join(opt.dataDir, opt.trainFile))
valid_defs, valid_tokens = count_defs(path.join(opt.dataDir, opt.validFile))
test_defs, test_tokens = count_defs(path.join(opt.dataDir, opt.testFile))
print('- building testing data...')
test_label_data = make_label_data(path.join(opt.dataDir, opt.testFile),
                   test_defs, test_tokens)
torch.save(path.join(opt.dataDir, opt.testLabelT7), test_label_data)
test_label_data = nil
print('- building validation data...')
valid_label_data = make_label_data(path.join(opt.dataDir, opt.validFile),
                  valid_defs, valid_tokens)
torch.save(path.join(opt.dataDir, opt.validLabelT7), valid_label_data)
valid_label_data = nil
print('- building training data...')
train_label_data = make_label_data(path.join(opt.dataDir, opt.trainFile),
                  train_defs, train_tokens)
torch.save(path.join(opt.dataDir, opt.trainLabelT7), train_label_data)
train_label_data = nil
