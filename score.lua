--[[Libraries]]--
-- neural network libraries
require 'dp'
require 'nn'
require 'optim'
require 'rnn'
require 'xlua'
-- Utilities
log = require 'log'
require 'eval.Perplexity'
require 'util.LMHelper'
require 'util.CharacterMap'
eval_util = require 'eval.eval'
gen_util = require 'util.gen'
--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Score definition entries')
cmd:text('Options:')
--[[ option ]]--
-- Data --
cmd:option('--batchSize', 64, 'number of examples per batch')
cmd:option('--dataDir', 'data/commondefs', 'dataset directory')
cmd:option('--dataType', 'sentence', 'sentence or labeled')
cmd:option('--skipSeed', false, 'for labeled data only')
cmd:option('--sosId', 2, 'start of sentence id')
cmd:option('--eosId', 1, 'end of sentence id')
cmd:option('--w2iFile', 'word2index.t7', 'word to index mapping')
cmd:option('--i2wFile', 'index2word.t7', 'index to word mapping')
-- Embedding --
cmd:option('--embeddingSize', 300, 'number of embedding units.')
cmd:option('--embFilepath', 'data/commondefs/auxiliary/emb.t7', 'path to word embedding torch binary file. See preprocess/prep_w2v.lua')
cmd:option('--hyperEmbFilepath', 'data/commondefs/auxiliary/hypernym_embs.t7', 'path to hypernym embeddings, a torch binary file. See preprocess/prep_hypernyms.lua')
-- Model --
cmd:option('--mode', 'sen', 'sen, ri')
cmd:option('--RICharCNN', false, 'enable character CNN')
cmd:option('--RIHypernym', false, 'enable hypernym embeddings')
cmd:option('--modelDir', 'data/commondefs/models/cur', 'path to load/save model and configuration')
cmd:option('--modelName', 'best_model.t7', 'path to load/save model and configuration')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--cudnnCNN', false, 'use CUDNN CNN implementation')
-- Reporting --
cmd:option('--entryFile', '', 'a file containing word<tab>definition per line')
cmd:option('--outputFile', 'score.txt', 'output file')
cmd:option('--outputGate', '', 'output for gate values')
cmd:option('--logFilepath', '', 'Log file path (std by default)')

cmd:text()
opt = cmd:parse(arg or {})

if opt.cuda then
  require 'cutorch'
  require 'cunn'
  if opt.cudnnCNN then
    require 'cudnn'
    cudnn.fastest = true
    cudnn.benchmark = true
  end
end
if opt.logFilepath == '' then
  opt.logFilepath = path.join(opt.modelDir, 'eval_log.txt')
end
log.outfile = opt.logFilepath
w2i = torch.load(path.join(opt.dataDir, opt.w2iFile))
i2w = torch.load(path.join(opt.dataDir, opt.i2wFile))
opt.numVocab = #i2w
opt.nSkips = opt.skipSeed and 2 or 0
log.info('Configurations: \n' .. table.tostring(opt, '\n'))
opt.w2i = w2i
opt.i2w = i2w
lm_factory = require 'model.lm'
if opt.RICharCNN then
  opt.charMap = CharacterMap(i2w)
end
if opt.RIHypernym then
  hlookup = nn.LookupTable(opt.numVocab, opt.embeddingSize)
  hlookup.maxOutNorm = -1
  local hemb = torch.load(opt.hyperEmbFilepath)
  hlookup.weight:copy(hemb)
end

helper = LMHelper(opt)

--[[Model]]--
log.info('Loading model...')
lm = torch.load(opt.modelDir .. '/best_model.t7')
lookup = lm_factory.get_lookup(opt)
if opt.mode ~= 'sen' then
  lookup_ri = lm_factory.get_lookup(opt)
  lm_factory.share_parameters(lookup, lookup_ri)
end
--[[Setting models for helper]]--
helper.lm = lm
helper.lookup = lookup
helper.lookup_ri = lookup_ri
helper.hlookup = hlookup
--[[Convert model to cuda]]--
if opt.cuda then
  lm:cuda()
  if lookup then lookup:cuda() end
  if lookup_ri then lookup_ri:cuda() end
  if hlookup then hlookup:cuda() end
end
log.info('Model:\n' .. lm:__tostring__())
--[[Reset state]]--
if opt.outputGate == '' then lm:evaluate()
else
  lm:training()
  local dropouts = lm_factory.find_modules(lm, 'nn.Dropout')
  for i = 1, #dropouts do
    dropouts[i].p = 0
  end
end
lm:forget()
collectgarbage()

local function compute_score(words, definitions)
  local batch = {}
  batch.new = true
  batch.x, batch.y, batch.label, batch.m = gen_util.entry2tensor(words, definitions, w2i)
  local predictions = helper:predict(batch)
  local ppls = Perplexity.sentence_ppl(predictions, batch.y, batch.m)
  return ppls, batch
end

local function write_score(ppls, ofp)
  for i =1, #ppls do
    ofp:write(ppls[i])
    ofp:write('\n')
  end
end

local function write_gate_values(input, ofp)
  if not ofp then return end
  local s = lm.modules[2].modules[1].modules[3]
  local clones = s.modules[1].sharedClones
  for i = 1, input:size(1) do
    for j = 1, input:size(2) do
      local word = i2w[input[i][j]]
      local gate_values = clones[j].modules[1].modules[1].output
      local z = gate_values[1][{{i}, {}}]
      local r = gate_values[2][{{i}, {}}]
      ofp:write(word)
      ofp:write('\n')
      for k = 1, z:size(2) do
        ofp:write(z[{1, k}])
        ofp:write(' ')
      end
      ofp:write('\n')
      for k = 1, r:size(2) do
        ofp:write(r[{1, k}])
        ofp:write(' ')
      end
      ofp:write('\n')
      if input[i][j] == opt.eosId then break end
    end
  end
end

if opt.entryFile ~= '' then
  local timer = torch.Timer()
  log.info('Calculating perplexity score for each entry...')
  local ofp = io.open(opt.outputFile, 'w')
  local ofp2
  if opt.outputGate ~= '' then ofp2 = io.open(opt.outputGate, 'w') end
  local words, definitions = {}, {}
  for line in io.lines(opt.entryFile) do
    local parts = stringx.split(line, '\t')
    local word = parts[1]
    local definition = stringx.split(parts[#parts], ' ')
    -- collect a batch
    table.insert(words, word)
    table.insert(definitions, definition)
    if #words == opt.batchSize then
      local pred, batch = compute_score(words, definitions)
      write_score(pred, ofp)
      write_gate_values(batch.y, ofp2)
      -- reset batch
      words, definitions = {}, {}
    end
  end
  if #words > 0 then
    local pred, batch = compute_score(words, definitions)
    write_score(pred, ofp)
    write_gate_values(batch.y, ofp2)
  end
  ofp:close()
  log.info(string.format('- Elapsed time = %ds', timer:time().real))
end
