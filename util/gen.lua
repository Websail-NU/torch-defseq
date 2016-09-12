local gen_util = {}

gen_util.table2words = function(i2w, ids, eos_id, delimiter)
  delimiter = delimiter and delimiter or ' '
  local sentences = {}
  for i = 1, #ids do
    local sentence = {}
    for j = 1, #ids[i] do
      if ids[i][j] == eos_id then break end
      sentence[j] = i2w[ids[i][j]]
    end
    sentences[i] = stringx.join(delimiter, sentence)
  end
  return sentences
end

gen_util.tensor2words = function(i2w, tensor, eos_id, delimiter)
  delimiter = delimiter and delimiter or ' '
  local sentences = {}
  for i = 1, tensor:size(1) do
    local sentence = {}
    for j = 1, tensor:size(2) do
      if tensor[i][j] == eos_id then break end
      sentence[j] = i2w[tensor[i][j]]
    end
    sentences[i] = stringx.join(delimiter, sentence)
  end
  return sentences
end

gen_util.ids2words = function(i2w, data, eos_id, delimiter)
  if torch.type(data) == 'table' then
    return gen_util.table2words(i2w, data, eos_id, delimiter)
  else
    return gen_util.tensor2words(i2w, data, eos_id, delimiter)
  end
end

gen_util.seed2tensor = function(seeds, w2i)
  local t_seeds = torch.IntTensor(#seeds, 3)
  local t_labels = torch.IntTensor(#seeds)
  local sos_id = w2i['<s>']
  local def_id = w2i['<def>']
  for i = 1,#seeds do
    local idx = w2i[seeds[i]]
    t_seeds[i][1] = sos_id
    t_seeds[i][2] = idx
    t_seeds[i][3] = def_id
    t_labels[i] = idx
  end
  return t_seeds, t_labels
end

gen_util.entry2tensor = function(words, definitions, w2i)
  local max_len = 0
  for i = 1,#definitions do
    if max_len < #definitions[i] then
      max_len = #definitions[i]
    end
  end
  local t_sentence = torch.IntTensor(#words, max_len + 4)
  local t_target = torch.IntTensor(#words, max_len + 4)
  local t_label = torch.IntTensor(#words)
  local t_mask = torch.IntTensor(#words, max_len + 4)
  local sos_id = w2i['<s>']
  local def_id = w2i['<def>']
  local eos_id = w2i['</s>']
  t_sentence:fill(eos_id)
  t_target:fill(eos_id)
  t_mask:fill(0)
  for i = 1, #words do
    local widx = w2i[words[i]]
    t_label[i] = widx
    t_sentence[i][1] = sos_id
    t_sentence[i][2] = widx
    t_sentence[i][3] = def_id
    t_mask[i][3] = 1
    for j = 1, #definitions[i] do
      t_sentence[i][j+3] = w2i[definitions[i][j]]
      t_mask[i][j + 3] = 1
    end
  end
  t_target[{{}, {1, max_len + 3}}]:copy(t_sentence[{{}, {2, max_len + 4}}])
  return t_sentence, t_target, t_label, t_mask
end

local function select_word(pred, sampling)
  local words
  if sampling then
    local probs = torch.exp(pred[#pred])
    words = torch.multinomial(probs, 1)
  else
    _, words = torch.max(pred[#pred], 2)
  end
  return words
end

local function basic_gen(seed_ids, helper, max_len, label_ids, sampling, n_sampling)
  max_len = max_len and max_len or 20
  n_sampling = n_sampling and n_sampling or 1
  if n_sampling > 1 then
    seed_ids = seed_ids:expand(n_sampling, 3)
    label_ids = label_ids:expand(n_sampling)
  end
  local batch = {new=true, x=seed_ids, label=label_ids}
  local pred = helper:predict(batch)
  local output = torch.IntTensor(seed_ids:size(1), max_len)
  batch.new = false
  for i = 1, max_len do
    local samples = select_word(pred, sampling)
    output[{{},i}]:copy(samples)
    batch.x = samples
    pred = helper:predict(batch)
  end
  return output
end

gen_util.sample = function(seed_ids, helper, max_len, label_ids, n_sampling)
  return basic_gen(seed_ids, helper, max_len, label_ids, true, n_sampling)
end

gen_util.greedy = function(seed_ids, helper, max_len, label_ids)
  return basic_gen(seed_ids, helper, max_len, label_ids, false)
end

--[ Local functions for beam search ]--

local function stage_lstm(beam, lstms)
  local T = lstms[1]._output:size(1)
  for i = 1, #lstms do
    lstms[i]._output[T]:copy(beam.h[i])
    lstms[i].cell[T]:copy(beam.c[i])
  end
end

local function get_lstm_states(lstms)
  local h_states, c_states = {}, {}
  local T = lstms[1]._output:size(1)
  for i = 1, #lstms do
    table.insert(h_states, lstms[i]._output[T]:clone())
    table.insert(c_states, lstms[i].cell[T]:clone())
  end
  return h_states, c_states
end

local function beam_state(score, prev_beam, idx, lstms)
  local copy_seq = {}
  local sum_score = score * (#prev_beam.seq + 1) -- change average to sum
  for k, v in pairs(prev_beam.seq) do copy_seq[k] = v end
  table.insert(copy_seq, idx)
  local h, c = get_lstm_states(lstms)
  return {score=sum_score, sc=score, seq=copy_seq, c=c, h=h}
end

local function update_beams(beam, pred, beams, k, lstms, max_len)
  local v = pred[#pred]:size(2)
  local nb = #beams
  local scores = pred[#pred].new():resize(v + nb) -- vocab + beams below b
  scores = scores:fill(beam.score)
  scores[{{1, v}}]:add(pred[#pred]):div(#beam.seq + 1) -- average
  for i = 1, nb do
    scores[v + i] = beams[i].sc
  end
  local sort_scores, map = torch.sort(scores, 1, true)
  local new_beams = {}
  for i =1, k do
    local idx = map[i]
    local score = sort_scores[i]
    if idx > v then
      table.insert(new_beams, beams[idx - v]) -- after vocab
    else
      table.insert(new_beams, beam_state(score, beam, idx, lstms))
    end
  end
  return new_beams
end

local function next_possible_beam(beams, max_len, eos_id)
  for i = 1,#beams do
    local beam = beams[i]
    local len = #beam.seq
    if len == 0 then
      return i
    end
    if len > 0 and len < max_len and beam.seq[len] ~= eos_id then
      return i
    end
  end
  return #beams + 1
end

gen_util.beam = function(seed_ids, helper, lstms, eos_id, k, max_len, label_ids)
  max_len = max_len and max_len or 20
  k = k and k or 5
  local batch = {new=true, x=seed_ids, label=label_ids}
  local pred = helper:predict(batch)
  local h, c = get_lstm_states(lstms)
  local beam = {score=0, seq={}, c=c, h=h}
  local beams = update_beams(beam, pred, {}, k, lstms, max_len)
  batch.new = false
  while true do
    local next_idx = next_possible_beam(beams, max_len, eos_id)
    if next_idx > #beams then break end
    local beam = table.remove(beams, next_idx)
    stage_lstm(beam, lstms)
    batch.x = torch.IntTensor{{beam.seq[#beam.seq]}}
    pred = helper:predict(batch)
    beams = update_beams(beam, pred, beams, k, lstms, max_len)
  end
  return beams
end

return gen_util
