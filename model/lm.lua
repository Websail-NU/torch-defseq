require 'dp'
require 'nn'
require 'rnn'
require 'dpnn'
include('JoinOneToMany.lua')

local lm_factory = {}
lm_factory.marker_name = '_marker_name_'

local function getLSTMConfig(config)
  config = config or {}
  local opt =
    xlua.unpack(
      {config},
      'lstm',
      'LSTM Language Model',
      -- embedding --
      {arg='embFilepath', type='string', default='',
       help='Path to word embedding. See preprocess/prep_w2v.lua.'},
      {arg='embeddingSize', type='number', default=300},
      {arg='embProjectSize', type='number', default=-1},
      -- LSTM --
      {arg='hiddenSizes', type='table', default={100},
       help='number of hidden units. i.e. {100,100}'},
      -- output --
      {arg='numVocab', type='number', default=29167},
      {arg='initLogitWithEmb', type='boolean', default=false},
      -- regularization --
      {arg='embDropoutProb', type='number', default=0.25},
      {arg='dropoutProb', type='number', default=0.5},
      {arg='mode', type='string', default='sen'},
      -- RI --
      {arg='RIProjectSize', type='number', default=-1},
      {arg='charMap', type='object', default=nil},
      {arg='cudnnCNN', type='boolean', default=false},
      {arg='charFeatureSizes', type='table', default={10, 30, 40, 40, 40}},
      {arg='charKernelSizes', type='table', default={2, 3, 4, 5, 6}},
      {arg='RIHypernym', type='boolean', default=false},
      {arg='RIMode', type='string', default='rnn'},
      -- misc
      {arg='initUniform', type='number', default=0.05,
       help='initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization'}
  )
  opt.LSTMInSize = opt.embeddingSize
  if opt.cudnnCNN then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
  end
  return opt
end

local function addEMBProjection(lm, opt)
  lm:add(nn.SplitTable(1))
  local project = nn.Sequential()
  project:add(nn.Linear(opt.embeddingSize, opt.embProjectSize))
  if opt.embDropoutProb > 0 then project:add(nn.Dropout(opt.embDropoutProb)) end
  project:add(nn.View(1, -1, opt.embProjectSize))
  lm:add(nn.Sequencer(project))
  lm:add(nn.JoinTable(1))
  opt.LSTMInSize = opt.embProjectSize
  return lm
end

local function addLSTM(lm, opt)
  local input_size = opt.LSTMInSize
  for i, hidden_size in ipairs(opt.hiddenSizes) do
    local lstm = nn.SeqLSTM(input_size, hidden_size)
    lm:add(lstm)
    if opt.dropoutProb > 0 then
      local dropout = nn.Dropout(opt.dropoutProb)
      lm:add(dropout)
    end
    input_size = hidden_size
  end
  opt.LSTMOutSize = input_size
  return lm
end

local function addCharCNN(opt)
  local max_len = opt.charMap.max_len
  local input_size = #opt.charMap.i2c
  local multiCNN = nn.ConcatTable()
  local cnn
  opt.CNNOutSize = 0
  for i = 1, #opt.charFeatureSizes do
    local act = nn.ReLU()
    local conv = nn.SpatialConvolution(1, opt.charFeatureSizes[i],
        input_size, opt.charKernelSizes[i], 1, 1, 0)
    local pool = nn.SpatialMaxPooling(1, max_len - opt.charKernelSizes[i] + 1)
    cnn = nn.Sequential():add(conv)
    cnn:add(act):add(pool):add(nn.Squeeze(3)):add(nn.Squeeze(3))
    if opt.cudnnCNN then cnn = cudnn.convert(cnn, cudnn) end
    multiCNN:add(cnn)
    opt.CNNOutSize = opt.CNNOutSize + opt.charFeatureSizes[i]
  end
  local v = nn.View(1, -1, input_size)
  v:setNumInputDims(2)
  local s = nn.Sequential():add(nn.OneHot(input_size)):add(v)
  if #opt.charFeatureSizes > 1 then s:add(multiCNN):add(nn.JoinTable(2))
  else s:add(cnn) end
  if opt.dropoutProb > 0 then s:add(nn.Dropout(opt.dropoutProb)) end
  return s
end

local function getRI(opt)
  local ri = nn.Sequential()
  opt.RIOutSize = 0
  if opt.embDropoutProb > 0 then
    ri:add(nn.Dropout(opt.embDropoutProb))
  else
    ri:add(nn.Identity())
  end
  if opt.RIProjectSize > 0 then
    ri:add(nn.Linear(opt.embeddingSize, opt.RIProjectSize))
    if opt.embDropoutProb > 0 then ri:add(nn.Dropout(opt.embDropoutProb)) end
    opt.RIOutSize = opt.RIOutSize + opt.RIProjectSize
  else
    opt.RIOutSize = opt.RIOutSize + opt.embeddingSize
  end
  local para
  if opt.charMap or opt.RIHypernym then
    para = nn.ParallelTable()
    para:add(ri)
  end
  if opt.charMap then
    local cnn = addCharCNN(opt)
    para:add(cnn)
    opt.RIOutSize = opt.RIOutSize + opt.CNNOutSize
  end
  if opt.RIHypernym then
    if opt.embDropoutProb > 0 then
      para:add(nn.Dropout(opt.embDropoutProb))
    else
      para:add(nn.Identity())
    end
    opt.RIOutSize = opt.RIOutSize + opt.embeddingSize
  end
  if opt.charMap or opt.RIHypernym then
    ri = nn.Sequential():add(para):add(nn.JoinTable(2))
  end
  return ri
end

local function getRIGRUGate(opt)
  -- input: table {RI, LSTM Output}
  -- output: table {t.*d, 1-t} (d is a candidate rep, 1-t is a carry gate)
  local split_gates = nn.ConcatTable()
    :add(nn.Narrow(2, 1, opt.LSTMOutSize))
    :add(nn.Sequential()
      :add(nn.Narrow(2, opt.LSTMOutSize + 1, opt.RIOutSize)) -- z
      :add(nn.Padding(2, opt.LSTMOutSize, 2, 1))) -- r
  local gates = nn.Sequential()
    :add(nn.Linear(opt.RIOutSize + opt.LSTMOutSize, opt.RIOutSize + opt.LSTMOutSize))
    :add(nn.Sigmoid()):add(split_gates)
  gates[lm_factory.marker_name] = 'gate'
  local output = nn.Sequential()
    :add(nn.ConcatTable():add(gates):add(nn.Identity()))
    :add(nn.FlattenTable()) -- {z, r, [ri;lstm]}
    :add(nn.ConcatTable()
      :add(nn.Sequential()
        :add(nn.SelectTable(1))
        :add(nn.ConcatTable()
          :add(nn.Sequential()
            :add(nn.MulConstant(-1, false)):add(nn.AddConstant(1, true)))
          :add(nn.Identity()))) -- {1-z, z}
      :add(nn.Sequential()
        :add(nn.NarrowTable(2,2)):add(nn.CMulTable())
        :add(nn.Linear(opt.RIOutSize + opt.LSTMOutSize, opt.LSTMOutSize))
        :add(nn.Tanh()))) -- new_h
    :add(nn.FlattenTable()):add(nn.ConcatTable() --{1-z, z, new_h}
      :add(nn.Sequential():add(nn.NarrowTable(2,2)):add(nn.CMulTable()))
      :add(nn.SelectTable(1))) -- {z.*new_h, 1-z}
  local tmp = nn.Sequential()
    :add(nn.JoinOneToMany()):add(nn.SplitTable(1)):add(nn.Sequencer(output))
    return tmp
end

local function attachPreRI(lm, ri, opt)
  local para = nn.ParallelTable()
    :add(ri):add(lm)
  local lm = nn.Sequential()
    :add(para):add(nn.JoinOneToMany())
  opt.LSTMInSize = opt.LSTMInSize + opt.RIOutSize
  return lm
end

local function attachPostRI(lm, ri, opt)
  local para = nn.ParallelTable()
    :add(ri):add(lm)
  if opt.RIMode == 'concat' then
    local project = nn.Sequential()
      :add(nn.Linear(opt.RIOutSize + opt.LSTMOutSize, opt.LSTMOutSize))
      :add(nn.Tanh())
    if opt.dropoutProb > 0 then
      project:add(nn.Dropout(opt.dropoutProb))
    end
    local lm = nn.Sequential()
      :add(para):add(nn.JoinOneToMany()):add(nn.SplitTable(1))
      :add(nn.Sequencer(project))
    return lm
  elseif opt.RIMode == 'gated' then
    local gatedRI = getRIGRUGate(opt)
    local lstmHighway = nn.Sequential()
      :add(nn.SelectTable(2)):add(nn.SplitTable(1))
      :add(nn.Sequencer(nn.Identity()))
    local carryLSTM = nn.Sequential()
      :add(nn.NarrowTable(2,2)):add(nn.CMulTable())
    local output = nn.Sequential()
      :add(nn.FlattenTable()):add(nn.ConcatTable()
        :add(nn.SelectTable(1))
        :add(carryLSTM))
        :add(nn.CAddTable())
    if opt.dropoutProb > 0 then
      output:add(nn.Dropout(opt.dropoutProb))
    end
    local lm = nn.Sequential()
      :add(para):add(nn.ConcatTable():add(gatedRI):add(lstmHighway))
      :add(nn.ZipTable()):add(nn.Sequencer(output))
    return lm
  end
  return lm
end

local function addOutput(lm, opt)
  local softmax = nn.Sequential()
  local logit = nn.Linear(opt.LSTMOutSize, opt.numVocab)
  softmax:add(logit)
  softmax:add(nn.LogSoftMax())
  lm:add(nn.Sequencer(softmax))
  opt.softmax_logit = logit
  return lm
end

-------------------------------------------
-------------------------------------------
-- Public functions
-------------------------------------------
-------------------------------------------

lm_factory.share_parameters = function(sharing, target)
  if torch.isTypeOf(target, 'nn.gModule') then -- for nngraph
    for i = 1, #target.forwardnodes do
      local node = target.forwardnodes[i]
      if node.data.module then
        lm_factory.share_parameters(
          sharing.forwardnodes[i].data.module,
          node.data.module)
      end
    end
  elseif torch.isTypeOf(target, 'nn.Module') then -- for normal node
    target:share(sharing, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('cannot share parameters of the argument type')
  end
end

lm_factory.get_lookup = function(opt)
  local lookup = nn.LookupTable(opt.numVocab, opt.embeddingSize)
  lookup.maxOutNorm = -1
  if opt.embFilepath ~= '' then
    local emb = torch.load(opt.embFilepath)
    lookup.weight:copy(emb)
  end
  return lookup
end

lm_factory.attach_lookup = function(lm, lookup)
  assert(torch.type(lm) == 'nn.Sequential')
  lm:insert(lookup, 1)
  return lm
end

lm_factory.create_model = function(config)
  local opt = getLSTMConfig(config)
  local lm = nn.Sequential()
  -- Input --
  if opt.embDropoutProb > 0 then lm:add(nn.Dropout(opt.embDropoutProb)) end
  if opt.embProjectSize > 0 then lm = addEMBProjection(lm, opt) end
  -- RI Input and attach pre RI--
  local ri
  if opt.mode == 'ri' then
    ri = getRI(opt)
    lm = opt.RIMode == 'rnn' and attachPreRI(lm, ri, opt) or lm
  end
  -- LSTM --
  lm = addLSTM(lm, opt)
  -- attach post RI --
  if opt.mode == 'ri' and opt.RIMode ~= 'rnn' then
    lm = attachPostRI(lm, ri, opt)
  else
    lm:add(nn.SplitTable(1))
  end
  -- Output --
  lm = addOutput(lm, opt)
  -- initialize parameters --
  if opt.initUniform > 0 then
    for k, params in ipairs(lm:parameters()) do
      params:uniform(-opt.initUniform, opt.initUniform)
    end
  end
  -- Lookup table --
  local lookup = lm_factory.get_lookup(opt)
  local lookup_ri
  if opt.mode ~= 'sen' then
    lookup_ri = lm_factory.get_lookup(opt)
    lm_factory.share_parameters(lookup, lookup_ri)
  end
  -- Init softmax logit wtih embedding
  if opt.initLogitWithEmb then
    opt.softmax_logit.weight:copy(lookup.weight)
  end
  return lm, lookup, lookup_ri
end

lm_factory.addTemperature = function(lm, T, cuda, use_softmax, output_both)
  local output = lm.modules[#lm.modules].module.module
  local logit
  for i = 1, #output.modules do
    if torch.type(output.modules[i]) == 'nn.Linear' then
      logit = output.modules[i]
    end
  end
  if not logit then
    assert(false, 'Cannot find logit layer')
  end
  local new_output = nn.Sequential():add(logit)
  if output_both then
    new_output:add(nn.ConcatTable()
      :add(nn.LogSoftMax())
      :add(nn.Sequential()
        :add(nn.MulConstant(1/T))
        :add(use_softmax and nn.SoftMax() or nn.LogSoftMax())))
  else
    new_output:add(nn.MulConstant(1/T))
      :add(use_softmax and nn.SoftMax() or nn.LogSoftMax())
  end
  new_output = nn.Sequencer(new_output)
  if cuda then new_output = new_output:cuda() end
  lm:remove(#lm.modules)
  lm:add(new_output)
  if output_both then
    lm:add(nn.ConcatTable()
      :add(nn.Sequencer(nn.SelectTable(1)))
      :add(nn.Sequencer(nn.SelectTable(2))))
  end
  return lm
end

local function match_module(module, module_name, marker_name)
  if marker_name then
    return module[lm_factory.marker_name] == marker_name
  else
    return torch.isTypeOf(module, module_name)
  end
end

local function rfind_modules(model, module_name, modules, marker_name)
  modules = modules and modules or {}
  if torch.isTypeOf(model, 'nn.Container') then
    for i = 1, #model.modules do
      if torch.isTypeOf(model.modules[i], 'nn.Container') then
        rfind_modules(model.modules[i], module_name, modules, marker_name)
      end
      if match_module(model.modules[i], module_name, marker_name) then
        table.insert(modules, model.modules[i])
      end
    end
  elseif match_module(model, module_name, marker_name) then
    table.insert(modules, model)
  end
end

lm_factory.find_modules = function(model, module_name, marker_name)
  local modules = {}
  rfind_modules(model, module_name, modules, marker_name)
  return modules
end

return lm_factory
