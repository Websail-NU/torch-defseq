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
    if batch.new then is_start = true end
    if is_start then table.insert(data, batch.x[1][1]) end
    if batch.x[1][1] == helper.eos_id and is_start then break end
  end
  if #data > 0 then
    local len = #data
    data = torch.IntTensor(data):reshape(1, len)
    if opt.cuda then data = data:cuda() end
    label = data[{{1}, 2}]
    correct_word = label[1]
    return data, label, correct_word
  end
  return nil, nil, nil
end

local function duplicate_data(data, label, size, mode)
  local new_data = torch.Tensor(size, data:size(2)):type(data:type())
  local new_label = torch.Tensor(size):type(label:type())
  for i = 1, size do
    new_data[{{i}, {}}]:copy(data)
  end
  new_label = new_data[{{}, 2}]
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
-- Functions for FWD
-------------------------------------------

local function collect_words_defs(filepath)
  local correct_defs = {}
  local test_word2index = {}
  local line_num = 1
  local word_count = 1
  for line in io.lines(filepath) do
    line = stringx.strip(line)
    local parts = stringx.split(line, '\t')
    local word = parts[1]
    if not test_word2index[word] then
      test_word2index[word] = word_count
      correct_defs[word] = {}
      word_count = word_count + 1
    end
    table.insert(correct_defs[word], line_num)
    line_num = line_num + 1
  end
  return test_word2index, correct_defs
end

local function collect_ppls(filepath, num_words, num_defs, test_word2index)
  local ppls = {}
  local line_num = 1
  local index_missing = false
  for line in io.lines(filepath) do
    line = stringx.strip(line)
    local parts = stringx.split(line, '\t')
    for i = 4, #parts, 2 do
      local idx = test_word2index[parts[i]]
      if not idx then
        idx = num_words + 1
        index_missing = true
      end -- this should not happen
      if not ppls[idx] then
        ppls[idx] = {}
      end
      ppls[idx][line_num] = tonumber(parts[i+1])
    end
    line_num = line_num + 1
  end
  if index_missing then
    log.warn(string.format('some words are not found in the test word index'))
  end
  local ppl_tensor = torch.Tensor(num_words, num_defs)
  ppl_tensor:fill(1e309)
  for i = 1, num_words do
    if ppls[i] then
      ppl_tensor[i][{{1, #ppls[i]}}] = torch.Tensor(ppls[i])
    end
  end
  return ppl_tensor
end

local function table2dict(t)
  d = {}
  for i = 1, #t do
    d[t[i]] = true
  end
  return d
end

-------------------------------------------
-------------------------------------------
-- Public functions
-------------------------------------------
-------------------------------------------

eval_util.ppl = function(helper, opt, ppl, ppl_by_len)
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
    if true then break end
    if ppl_by_len then
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
  assert(not opt.HSM) -- not yet support
  helper.lm:evaluate()
  local ppls = torch.Tensor(2, #indexes)
  local sum_rank = 0
  local top1 = 0
  local top10 = 0
  local top100 = 0
  local count = 0
  while true do
    local data, label, correct_word = collect_data(helper, opt)
    if not data then break end
    count = count + 1
    ppls:zero()
    data, label = duplicate_data(data, label, opt.batchSize, opt.mode)
    local start_ll_index = 3
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
        -- at i position, we are predicting data at i+1 position
        -- thus start at 3 (input: <def>, output: first word of def)
        --      end at len - 1 (input: last word of def, output: </s>)
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

eval_util.fwd = function(filepath, num_words, num_defs)
  local test_word2index, correct_defs = eval_util._collect_words_defs(filepath)
  local ppl_tensor = eval_util._collect_ppls(filepath, num_words, num_defs, test_word2index)
  local sum_rank = 0
  local top1 = 0
  local top10 = 0
  local top100 = 0
  local w_count = 0
  local d_count = 0
  for w, i in pairs(test_word2index) do
    local correct = eval_util.table2dict(correct_defs[w])
    local ppl = ppl_tensor[i]
    local _, index = torch.sort(ppl)
    local rank = index:size(1)
    for j = 1, index:size(1) do
      if correct[index[j]] then
        sum_rank = sum_rank + j
        d_count = d_count + 1
        rank = math.min(j, rank)
      end
    end
    if rank == 1 then top1 = top1 + 1 end
    if rank <= 10 then top10 = top10 + 1 end
    if rank <= 100 then top100 = top100 + 1 end
    w_count = w_count + 1
  end
  return sum_rank/d_count, top1/w_count, top10/w_count, top100/w_count, ppl_tensor
end

return eval_util
