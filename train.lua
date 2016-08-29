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

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a LSTM langauge model on definition dataset.')
cmd:text('Options:')
-- Data --
cmd:option('--dataDir', 'data/commondefs', 'dataset directory')
cmd:option('--modelDir', 'data/commondefs/models/cur', 'path to load/save model and configuration')
cmd:option('--dataType', 'sentence', 'sentence or labeled')
cmd:option('--skipSeed', false, 'for labeled data only')
cmd:option('--sosId', 2, 'start of sentence id')
cmd:option('--eosId', 1, 'end of sentence id')
-- Training options --
cmd:option('--optim', 'sgd', 'Optimization methods (i.e. sgd, adam, adagrad, rmsprop, ...)')
cmd:option('--learningRate', 1.0, 'learning rate at t=0')
cmd:option('--lrMomentum', 0, 'learning rate momentum')
cmd:option('--lrDecayEvery', -1, 'number of epochs before learning rate is decayed')
cmd:option('--lrDecayPPLImp', 0.96, 'improvement ratio between val ppl before decaying learning rate')
cmd:option('--lrDecayPPLWait', 2, 'number of non-improving epochs to wait before decaying learning rate')
cmd:option('--lrDecayFactor', 0.8, 'factor by which learning rate is decayed (lr = lr * factor)')
cmd:option('--gradClip', 5, 'clamp gradients of each batch (-gradClip >=b grad >= gradClip)')
cmd:option('--batchSize', 64, 'number of examples per batch')
cmd:option('--rho', 10, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--cudnnCNN', false, 'use CUDNN CNN implementation')
-- Embedding --
cmd:option('--embeddingSize', 300, 'number of embedding units.')
cmd:option('--embProjectSize', -1, 'non-linear projection for embedding (Sentence)')
cmd:option('--initLogitWithEmb', false, 'initialize logit with the embedding.')
cmd:option('--embFilepath', 'data/commondefs/auxiliary/emb.t7', 'path to word embedding torch binary file. See preprocess/prep_w2v.lua')
cmd:option('--hyperEmbFilepath', 'data/commondefs/auxiliary/hypernym_embs.t7', 'path to hypernym embeddings, a torch binary file. See preprocess/prep_hypernyms.lua')
-- Model --
cmd:option('--hiddenSizes', '{300}', 'number of hidden units. i.e. {100, 100}')
cmd:option('--mode', 'sen', 'sen, ri')
cmd:option('--RIMode', 'rnn', 'rnn, concat, gated')
cmd:option('--RIProjectSize', -1, 'linear projection for embedding (RI)')
cmd:option('--RICharCNN', false, 'enable character CNN')
cmd:option('--RIHypernym', false, 'enable hypernym embeddings')
-- Regularization --
cmd:option('--embDropoutProb', 0.25, 'embedding dropout probability')
cmd:option('--dropoutProb', 0.5, 'dropout probability')
-- Misc --
cmd:option('--initUniform', 0.05, 'initialize parameters using uniform distribution')
cmd:option('--numVocab', 29167, 'vocabulary size of the dataset')
-- Reporting --
cmd:option('--saveAll', false, 'save all models')
cmd:option('--printSteps', 1000, 'number of steps (mini-batches) before printing our average loss')
cmd:option('--realTrainPPL', false, 'calcuate an accurate (masked) perplexity during training')
cmd:option('--logFilepath', '', 'Log file path (std by default)')

cmd:text()
opt = cmd:parse(arg or {})
opt.hiddenSizes = dp.returnString(opt.hiddenSizes)
--[[Additional libraries and options]]--
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
  opt.logFilepath = path.join(opt.modelDir, 'training_log.txt')
end
opt.nSkips = opt.skipSeed and 2 or 0
log.outfile = opt.logFilepath
log.info('Configurations: \n' .. table.tostring(opt, '\n'))
lm_factory = require 'model.lm'

--[[Additional data]]
if opt.RICharCNN then
  i2w = torch.load(path.join(opt.dataDir, 'index2word.t7'))
  opt.charMap = CharacterMap(i2w)
end
if opt.RIHypernym then
  hlookup = nn.LookupTable(opt.numVocab, opt.embeddingSize)
  hlookup.maxOutNorm = -1
  local hemb = torch.load(opt.hyperEmbFilepath)
  hlookup.weight:copy(hemb)
end

--[[Model: load existing if exists]]--
if LMHelper.fileExists(opt.modelDir .. '/latest.t7')
   and LMHelper.fileExists(opt.modelDir .. '/training_state.t7') then
  log.info('Resume training...')
  lm = torch.load(opt.modelDir .. '/latest.t7')
  lookup = lm_factory.get_lookup(opt)
  if opt.mode ~= 'sen' then
    lookup_ri = lm_factory.get_lookup(opt)
    lm_factory.share_parameters(lookup, lookup_ri)
  end
  state = torch.load(opt.modelDir .. '/training_state.t7')
  --log.info(table.tostring(state, '\n'))
else
  log.info('No previous model, creating new model...')
  lm, lookup, lookup_ri = lm_factory.create_model(opt)
  state = {
    optim_config = {learningRate=opt.learningRate,
                    momentum=opt.lrMomentum},
    steps = 0, best_val_ppl = math.huge, best_epoch = 0, start_epoch = 1,
    last_imp_val_ppl = math.huge, last_imp_epoch = 0, imp_wait = 0
  }
  if LMHelper.fileExists(opt.logFilepath) then
    os.remove(opt.logFilepath)
    local charMap = opt.charMap
    opt.charMap = nil
    log.info('Configurations: \n' .. table.tostring(opt, '\n'))
    opt.charMap = charMap
  end
end

helper = LMHelper(opt)

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
print('model loaded')
log.info('Model:\n' .. lm:__tostring__())

--[[Loss]]--
crit = LMHelper.SeqClassNNLCriterion(opt.HSM)
if opt.cuda then
  crit:cuda()
end

--[[Training: States]]--
ppl = Perplexity()
params, grad_params = lm:getParameters()
batch = {}
local timer = torch.Timer()
-- local words_per_step = opt.batchSize * opt.rho
local epoch_steps = 0
local epoch_words = 0

--[[Training: one batch]]--
function f(w)
  assert(w == params)
  grad_params:zero() -- reset gradients
  -- get data (batch is set in the main loop)
  local num_words = batch.m:sum()
  local batch_scaling = batch.m:size(1) / num_words
  --forward
  local predictions = helper:predict(batch)
  local loss = crit:forward(predictions, helper.targets)
  loss = loss / batch.x:size(2)
  -- backward
  local grad_predictions = crit:backward(predictions, helper.targets)
  helper:maskGradPred(grad_predictions)
  helper:dlm(grad_predictions)
  -- reporting performance
  if opt.realTrainPPL then
    ppl:add(predictions, helper.targets, helper.m)
  else -- Approximate perplexity from loss (no mask)
    local all_tokens = batch.m:size(2) * batch.m:size(1)
    ppl:addNLL(loss * all_tokens, all_tokens)
  end
  epoch_words = epoch_words + num_words
  grad_params:mul(batch_scaling)
  -- gradient clip
  if opt.gradClip > 0 then
    grad_params:clamp(-opt.gradClip, opt.gradClip)
  end
  return loss, grad_params
end

--[[Training: helper functions]]--
function reset()
  -- reset everything
  epoch_steps = 0
  epoch_words = 0
  helper.loader:reset('train')
  helper.loader:reset('val')
  lm:forget()
  lm:training()
  ppl:reset()
  timer:reset()
end

function updateLR(epoch, val_ppl)
  -- decay learning rate
  local old_lr = state.optim_config.learningRate
  if opt.lrDecayEvery > 0 and epoch % opt.lrDecayEvery == 0 then
    log.info('- Scheduled learning rate decay')
    state.optim_config.learningRate = old_lr * opt.lrDecayFactor
  elseif opt.lrDecayPPLImp > 0 then
    if val_ppl / state.last_imp_val_ppl < opt.lrDecayPPLImp then
      log.info('- Significant improvement found')
      log.info(string.format('-- ppl: %f -> %f', state.last_imp_val_ppl, val_ppl))
      log.info(string.format('-- ep: %d -> %d', state.last_imp_epoch, epoch))
      state.last_imp_val_ppl = val_ppl
      state.last_imp_epoch = epoch
      state.imp_wait = 0
    else
      state.imp_wait = state.imp_wait + 1
      log.info(string.format('- No significant improvement'))
      log.info(string.format('-- last improved since ep: %d', state.last_imp_epoch))
      log.info(string.format('-- waiting for %d eps', state.imp_wait))
    end
    if state.imp_wait >= opt.lrDecayPPLWait then
      log.info('- No improvement, learning rate decay')
      state.optim_config.learningRate = old_lr * opt.lrDecayFactor
      state.imp_wait = 0
      if opt.lrDecayFactor == 1.0 then return true end
    end
  end
  return false
end

--[[Training: Main Loop]]--
lm:remember('both') -- It is important to keep the state from batch to batch
lm:training()
collectgarbage()
for epoch = state.start_epoch, opt.maxEpoch do
  log.info(string.format('====================Start epoch %d====================', epoch))
  log.info(string.format('- Learning rate = %f', state.optim_config.learningRate))
  log.info('- Training...')
  reset()
  -- mini batch routine
  while true do
    batch = helper.loader:nextBatch('train')
    if not batch then break end
    local _, loss = optim[opt.optim](f, params, state.optim_config)
    state.steps = state.steps + 1
    epoch_steps = epoch_steps + 1
    if state.steps % opt.printSteps == 0 then
      log.info(string.format('-- Progress: %d@%d, loss: %f, ppl: %f, rate: %f wps',
            state.steps, epoch, loss[1], ppl:perplexity(),
            epoch_words / timer:time().real))
    end
  end
  -- end of epoch routine
  local train_ppl = ppl:perplexity()
  -- validation routine
  lm:evaluate()
  lm:forget()
  ppl:reset()
  log.info('- Validating...')
  opt.pplSplit = 'val'
  eval_util.ppl(helper, opt, ppl)
  local val_ppl = ppl:perplexity()
  log.info(string.format('- Train ppl = %f, val ppl = %f', train_ppl, val_ppl))
  -- save and update learning rate
  local done_training = updateLR(epoch, val_ppl)
  helper:clearModelState()
  if opt.saveAll then
    torch.save(opt.modelDir .. '/model_ep' .. epoch .. '.t7', lm)
    torch.save(opt.modelDir .. '/state_ep' .. epoch .. '.t7', state)
  end
  if val_ppl < state.best_val_ppl then
    log.info(string.format('- Best model updated'))
    log.info(string.format('-- ppl: %f -> %f', state.best_val_ppl, val_ppl))
    log.info(string.format('-- ep: %d -> %d', state.best_epoch, epoch))
    state.best_val_ppl = val_ppl
    state.best_epoch = epoch
    torch.save(opt.modelDir .. '/best_model.t7', lm)
  end
  state.start_epoch = epoch + 1
  torch.save(opt.modelDir .. '/latest.t7', lm)
  torch.save(opt.modelDir .. '/training_state.t7', state)
  log.info(string.format('- Epoch time: %ds', timer:time().real))
  log.info(string.format('- Best model ppl = %f @ ep = %d',
                          state.best_val_ppl, state.best_epoch))
  if done_training then break end
end
