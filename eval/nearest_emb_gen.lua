require 'dp'
log = require 'log'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a LSTM langauge model on definition dataset.')
cmd:text('Options:')
-- Data --
cmd:option('--dataDir', 'data/commondefs', 'dataset directory')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--embFilepath', 'data/commondefs/auxiliary/emb.t7',
           'path to word embedding torch binary file. See preprocess/prep_w2v.lua')
cmd:option('--outputFile', 'models/nearest/test_nearest.txt', 'file to save output')
cmd:option('--exampleFile', 'train.txt', 'data to copy definitions from')
cmd:option('--wordListFile', 'shortlist/shortlist_test.txt', 'words to generate definitions')
-- Reporting --
cmd:option('--logFilepath', '', 'Log file path (std by default)')

cmd:text()
opt = cmd:parse(arg or {})

w2i = torch.load(path.join(opt.dataDir, 'word2index.t7'))
i2w = torch.load(path.join(opt.dataDir, 'index2word.t7'))
emb = torch.load(opt.embFilepath)
if opt.cuda then
  require 'cutorch'
  emb = emb:cuda()
end
examples = {}
definitions = {}
dictionaries = {}
for line in io.lines(path.join(opt.dataDir, opt.exampleFile)) do
  local parts = stringx.split(line, '\t')
  local word = parts[1]
  local dict = parts[3]
  local definition = parts[4]
  dictionaries[dict] = true
  if not definitions[word] then definitions[word] = {} end
  table.insert(definitions[word], {dict, definition})
  examples[w2i[word]] = true
end
dict_ofp = {}
for k, v in pairs(dictionaries) do
  dict_ofp[k] = io.open(path.join(opt.dataDir, opt.outputFile..'.'..k), 'w')
end

ofp = io.open(path.join(opt.dataDir, opt.outputFile), 'w')
ofp2 = io.open(path.join(opt.dataDir, opt.outputFile..'.all'), 'w')
for word in io.lines(path.join(opt.dataDir, opt.wordListFile)) do
  local widx = w2i[word]
  local wemb = emb[widx]
  local dist = emb * wemb
  local v, idx = torch.sort(dist, true)
  local max_example_idx
  for i = 1,idx:size(1) do
    if examples[idx[i]] then
      max_example_idx = idx[i]
      break
    end
  end
  local nearest_defs = definitions[i2w[max_example_idx]]
  for i = 1, #nearest_defs do
    ofp:write(word)
    ofp:write('\t')
    ofp:write(nearest_defs[i][2])
    ofp:write('\n')
    ofp2:write(word)
    ofp2:write('\t')
    ofp2:write(' ')
    ofp2:write('\t')
    ofp2:write(nearest_defs[i][1])
    ofp2:write('\t')
    ofp2:write(nearest_defs[i][2])
    ofp2:write('\n')
    dict_ofp[nearest_defs[i][1]]:write(word)
    dict_ofp[nearest_defs[i][1]]:write('\t')
    dict_ofp[nearest_defs[i][1]]:write(nearest_defs[i][2])
    dict_ofp[nearest_defs[i][1]]:write('\n')
  end
end
ofp:close()
ofp2:close()
for k, v in pairs(dictionaries) do
  dict_ofp[k]:close()
end
