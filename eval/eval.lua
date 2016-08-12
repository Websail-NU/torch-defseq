require 'pl'
if not log then log = require 'log' end
local eval_util = {}

-------------------------------------------
-- Functions for RVD
-------------------------------------------

local function collect_data(helper, opt)
  local data = {}
  local label, correct_word
  local is_start = false
  while true do
    local batch = helper.loader:nextBatch('test')
    if not batch then break end
    if batch.new then
      is_start = true
      label = batch.label
    end
    if is_start then table.insert(data, batch.x[1][1]) end
    if batch.x[1][1] == helper.eos_id and is_start then break end
  end
  if #data > 0 then
    local len = #data
    data = torch.IntTensor(data):reshape(1, len)
    if opt.cuda then data = data:cuda() end
    if not opt.skipSeed then label = data[{{1}, 2}] end
    correct_word = label[1]
    return data, label, correct_word
  end
  return nil, nil, nil
end

local function duplicate_data(data, label, opt)
  local size = opt.batchSize
  local new_data = data.new():resize(size, data:size(2))
  local new_label = label.new():resize(size)
  for i = 1, size do
    new_data[{{i}, {}}]:copy(data)
    new_label[{{i}}]:copy(label)
  end
  if not opt.skipSeed then new_label = new_data[{{}, 2}] end
  return new_data, new_label
end

local function process_ppls_word_def(i2w, word, def, ppls, ofp)
  local _, sort_i = torch.sort(ppls[2])
  local rank = -1
  for j = 1, sort_i:size(1) do
    if ppls[{{}, sort_i[j]}][1] == word then
      rank = j
      break
    end
  end
  if ofp then
    ofp:write(i2w[word])
    ofp:write('\t')
    local def_words = {}
    for i = 1,def:size(1) do
      table.insert(def_words, i2w[def[i]])
    end
    ofp:write(stringx.join(' ', def_words))
    ofp:write('\t')
    ofp:write(rank)
    ofp:write('\t')
    for j = 1, sort_i:size(1) do
      ofp:write(i2w[ppls[{{}, sort_i[j]}][1]])
      ofp:write('\t')
      ofp:write(ppls[{{}, sort_i[j]}][2])
      ofp:write('\t')
    end
    ofp:write('\n')
  end
  return rank
end

-------------------------------------------
-------------------------------------------
-- Public functions
-------------------------------------------
-------------------------------------------

eval_util.ppl = function(helper, opt, ppl)
  ppl:reset()
  local batch = {}
  while true do
    batch = helper.loader:nextBatch(opt.pplSplit)
    if not batch then break end
    local predictions = helper:predict(batch)
    if batch.new and not opt.skipSeed then
      batch.m[{{},{1,2}}] = 0 -- also mask the first 2 tokens (w_d and <def>)
    end
    ppl:add(predictions, batch.y, batch.m)
    if opt.pplByLen then
      local sen_len = helper.loader.split_batch_mask[opt.pplSplit]:sum(2) - 1
      sen_len = sen_len:squeeze()
      for bidx = 1, opt.batchSize do
        local b_len = sen_len[bidx]
        if not ppl_by_len[b_len] then
          ppl_by_len[b_len] = Perplexity()
          ppl_by_len[b_len]:reset()
        end
        ppl_by_len[b_len]:addFromBatchIndex(predictions, batch.y, batch.m, bidx)
      end
    end
  end
end

eval_util.rvd = function(helper, opt, indexes, ofp)
  helper.lm:evaluate()
  local ppls = torch.Tensor(2, #indexes)
  local sum_rank, top1, top10, top100, count = 0,0,0,0,0
  local start_ll_index = opt.skipSeed and 1 or 3
  while true do
    local data, label, correct_word = collect_data(helper, opt)
    if not data then break end
    count = count + 1
    ppls:zero()
    data, label = duplicate_data(data, label, opt)
    for idx = 1, #indexes, opt.batchSize do
      local midx = math.min(idx+opt.batchSize-1, #indexes)
      local lidx = 1
      for sidx = idx, midx do
          label[lidx] = indexes[sidx] -- change the word
          lidx = lidx + 1
      end
      local batch = {x=data, label=label, new=true}
      local predictions = helper:predict(batch)
      lidx = 1
      for sidx= idx, midx do
        local ll = 0
        for i = start_ll_index, (data:size(2) - 1) do
          ll = ll + predictions[i][lidx][data[lidx][i+1]]
        end
        local ppl = torch.exp(-ll/(data:size(2) - start_ll_index))
        ppls[1][sidx] = indexes[sidx]
        ppls[2][sidx] = ppl
        lidx = lidx + 1
      end
    end
    local rank = process_ppls_word_def(opt.i2w,
      correct_word, data[1][{{start_ll_index + 1,-2}}], ppls, ofp)
    sum_rank = sum_rank + rank
    if rank == 1 then top1 = top1 + 1 end
    if rank <= 10 then top10 = top10 + 1 end
    if rank <= 100 then top100 = top100 + 1 end
  end
  return sum_rank/count, top1/count, top10/count, top100/count
end

return eval_util
