--[[Libraries]]--
-- neural network libraries
require 'dp'
require 'nn'
require 'optim'
require 'rnn'
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
cmd:text('Test a LSTM langauge model on definition dataset.')
cmd:text('Options:')
--[[ option ]]--
-- Data --
cmd:option('--batchSize', 64, 'number of examples per batch')
cmd:option('--dataDir', 'data/commondefs', 'dataset directory')
cmd:option('--modelDir', 'data/commondefs/models/cur', 'path to load/save model and configuration')
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
cmd:option('--ppl', false, 'compute PPL')
cmd:option('--pplSplit', 'test', 'data split to compute perplexity')
cmd:option('--pplByLen', false, 'Evaluate perplexity by length')
cmd:option('--gen', false, 'generate definition')
cmd:option('--temperature', 1, 'temperature')
cmd:option('--genWords', 'shortlist/shortlist_test.txt', 'file containing list of words.')
cmd:option('--genMethod', 'greedy', 'greedy, sampling, beam')
cmd:option('--genOutFile', 'gen.txt', 'output generanted definition file')
cmd:option('--genMaxLen', 30, 'number of tokens to generate')
cmd:option('--genSamples', 1, 'number of samples for sampling methods')
cmd:option('--beamWidth', 10, 'number top-k to sequences in beam search')
cmd:option('--rvd', false, 'compute reversed dictionary output')
cmd:option('--rvdWords', 'shortlist/shortlist_test.txt',
           'filename to a list of words in test set.')
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
lm:evaluate()
lm:forget()
collectgarbage()
--[[Perplexity]]--
if opt.ppl then
  local timer = torch.Timer()
  log.info('Calculating perplexity...')
  local ppl = Perplexity()
  local ppl_by_len = opt.pplByLen and {} or nil
  ppl:reset()
  helper.loader:reset(opt.pplSplit)
  eval_util.ppl(helper, opt, ppl, ppl_by_len)
  log.info(string.format('Perplexity:'))
  log.info(string.format('- Tokens = %d', ppl.samples))
  log.info(string.format('- LL = %f', ppl.ll))
  log.info(string.format('- PPL = %f', ppl:perplexity()))
  if opt.pplByLen then
    log.info(string.format('- PPL by length: '))
    for i, ippl in pairs(ppl_by_len) do
      log.info(string.format('-- len %d PPL = %f (samples = %d)',
              i, ippl:perplexity(), ippl.samples))
    end
  end
  log.info(string.format('- Elapsed time = %ds', timer:time().real))
end

--[[Generate definitions]]--
if opt.gen then
  local timer = torch.Timer()
  local ofp = io.open(path.join(opt.modelDir, opt.genOutFile), 'w')
  helper.lm = lm_factory.addTemperature(helper.lm, opt.temperature,
                                        opt.cuda, false, false)
  log.info('Generating definitions...')
  for line in io.lines(path.join(opt.dataDir, opt.genWords)) do
    local seed_ids, label_ids = gen_util.seed2tensor({line}, opt.w2i)
    if opt.mode == sen then label_ids = nil end
    local candidates
    if opt.genMethod == 'sampling' then
      candidates = {}
      local samples = gen_util.sample(
        seed_ids, helper, opt.genMaxLen, label_ids, opt.genSamples)
      for i = 1, opt.genSamples do
        table.insert(candidates, samples[{{i}, {}}])
      end
    elseif opt.genMethod == 'greedy' then
      candidates = {gen_util.greedy(seed_ids, helper, opt.genMaxLen, label_ids)}
    elseif opt.genMethod == 'beam' then
      local lstms = lm_factory.find_modules(helper.lm, 'nn.SeqLSTM')
      local beams = gen_util.beam(seed_ids, helper,
                                  lstms, opt.eosId, opt.beamWidth,
                                  opt.genMaxLen, label_ids)
      candidates = {}
      for i = 1, #beams do table.insert(candidates, beams[i].seq) end
    end
    for i = 1, #candidates do
      local sentences = gen_util.ids2words(opt.i2w, candidates[i], opt.eosId)
      ofp:write(line)
      ofp:write('\t')
      ofp:write(sentences[1])
      ofp:write('\n')
    end
  end
  ofp:close()
  log.info(string.format('- Elapsed time = %ds', timer:time().real))
end

--[[Rank words given a definition (Reversed dictionary)]]--
if opt.rvd then
  local timer = torch.Timer()
  log.info('Running reversed dictionary task...')
  local indexes = {}
  for line in io.lines(path.join(opt.dataDir, opt.rvdWords)) do
    line = stringx.strip(line)
    table.insert(indexes, w2i[line])
  end
  -- reset batch size and rho to 1
  helper.loader.split_batch_size['test'] = 1
  helper.loader.seq_length = 1
  helper.loader:reset('test')
  local ofp = io.open(path.join(opt.modelDir, 'rvd.out.tsv'), 'w')
  -- each line is a TSV start with word, definition, and rank followed by words and ppls
  local avg, t1, t10, t100 = eval_util.rvd(helper, opt, indexes, ofp)
  log.info(string.format('RVD:'))
  log.info(string.format('- Accuracy = %f', t1))
  log.info(string.format('- Top-10 Accuracy = %f', t10))
  log.info(string.format('- Top-100 Accuracy = %f', t100))
  log.info(string.format('- Average Rank = %f', avg))
  ofp.close()
  log.info(string.format('- Elapsed time = %ds', timer:time().real))
end

function file_exists(name)
  local f=io.open(name,"r")
  if f~=nil then io.close(f) return true else return false end
end
