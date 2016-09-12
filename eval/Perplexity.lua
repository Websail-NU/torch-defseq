local Perplexity = torch.class('Perplexity')

function Perplexity:__init()
  -- config = config or {sos_id=-1, eos_id=-1}
  self.ll = 0
  self.samples = 0
  -- self.sos_id = config.sos_id
  -- self.eos_id = config.eos_id
end

function Perplexity:reset()
  self.ll = 0
  self.samples = 0
end

function Perplexity:perplexity()
  return torch.exp(-self.ll / self.samples)
end

function Perplexity:addNLL(nll, samples)
  self.ll = self.ll + (nll * -1)
  self.samples = self.samples + samples
end

function Perplexity:add(predictions, targets, mask)
  if torch.type(predictions[1]) =='table' then
    self:_addHSM(predictions, mask)
  else
    self:_add(predictions, targets, mask)
  end
end

function Perplexity:_add(predictions, targets, mask)
  -- predictions: table of tensor (batch x vocab) or transpose
  -- targets: tensor of targets (batch x seq_len)
  -- mask: optional tensor of mask 1 or 0 (batch x seq_len)
  for i = 1, #predictions do -- seq
    local itargets
    if targets:size(1) == mask:size(1) then
      itargets = targets[{{},i}]
    else
      itargets = targets[{i,{}}]
    end
    local ipredictions = predictions[i]
    for j = 1, itargets:size(1) do -- batch
      local target_id = itargets[j]
      local m = mask and mask[{j, i}] or 1
      -- if target_id ~= self.sos_id and target_id ~= self.eos_id then
      if target_id ~= 0 then
        self.ll = self.ll + ipredictions[{j, target_id}] * m
        self.samples = self.samples + 1 * m
      end
    end
  end
end

function Perplexity:_addHSM(predictions, mask)
  -- predictions: table of tensor (batch x vocab) or transpose
  -- mask: optional tensor of mask 1 or 0 (batch x seq_len)
  for i = 1, #predictions do -- seq
    local cluster_logprob = predictions[i][1]
    local class_logprob = predictions[i][2]
    local c_target = predictions[i][3]
    for j = 1, c_target:size(1) do -- batch
      local m = mask and mask[{j, i}] or 1
      if target_id ~= 0 then
        self.ll = self.ll + cluster_logprob[{j, c_target[{j, 1}]}] * m
        self.ll = self.ll + class_logprob[{j, c_target[{j, 2}]}] * m
        self.samples = self.samples + 1 * m
      end
    end
  end
end

function Perplexity:addFromBatchIndex(predictions, targets, mask, batch_idx)
  if torch.type(predictions[1]) =='table' then
    assert(false, 'Not implemented')
  else
    self:_addBatchIdx(predictions, targets, mask, batch_idx)
  end
end

function Perplexity:_addBatchIdx(predictions, targets, mask, batch_idx)
  -- predictions: table of tensor (batch x vocab) or transpose
  -- targets: tensor of targets (batch x seq_len)
  -- mask: optional tensor of mask 1 or 0 (batch x seq_len)
  for i = 1, #predictions do -- seq
    local itargets
    if targets:size(1) == mask:size(1) then
      itargets = targets[{{},i}]
    else
      itargets = targets[{i,{}}]
    end
    local ipredictions = predictions[i]
    local target_id = itargets[batch_idx]
    local m = mask and mask[{batch_idx, i}] or 1
    if target_id ~= 0 then
      self.ll = self.ll + ipredictions[{batch_idx, target_id}] * m
      self.samples = self.samples + 1 * m
    end
  end
end

function Perplexity.sentence_ppl(predictions, targets, mask)
  -- predictions: table of tensor (batch x vocab) or transpose
  -- targets: tensor of targets (batch x seq_len)
  -- mask: optional tensor of mask 1 or 0 (batch x seq_len)
  local ll = torch.Tensor(mask:size(1)):fill(0)
  local samples = torch.IntTensor(mask:size(1)):fill(0)
  for i = 1, #predictions do -- seq
    local itargets
    if targets:size(1) == mask:size(1) then
      itargets = targets[{{},i}]
    else
      itargets = targets[{i,{}}]
    end
    local ipredictions = predictions[i]
    for j = 1, itargets:size(1) do -- batch
      local target_id = itargets[j]
      local m = mask and mask[{j, i}] or 1
      if target_id ~= 0 then
        ll[j] = ll[j] + ipredictions[{j, target_id}] * m
        samples[j] = samples[j] + 1 * m
      end
    end
  end
  local sen_ppl = {}
  for i = 1, ll:size(1) do
    table.insert(sen_ppl, torch.exp(-ll[i] / samples[i]))
  end
  return sen_ppl
end
