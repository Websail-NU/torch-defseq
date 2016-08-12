require 'pl'
require 'torch'
require 'dp'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Compute embeddings of the hypernyms of word.')
cmd:option('--dataDir', 'data/commondefs', 'Data directory')
cmd:option('--hypernymFile', 'auxiliary/bag_of_hypernyms.txt', 'hypernyms of words')
cmd:option('--w2v', '/websail/common/embeddings/word2vec/GoogleNews-vectors-negative300.t7', 'Torch binary file of Mikolov word2vec, see word2vec.torch for conversion.')
cmd:option('--w2i', 'word2index.t7', 'Input map from word to index in torch binary file')
cmd:option('--i2w', 'index2word.t7', 'Input map from index to word in torch binary file')
cmd:option('--embT7', 'auxiliary/hypernym_embs.t7', 'Output embeddings')
cmd:option('--limit', 5, 'Number of hypernyms to average')
cmd:option('--limitVocab', '', 'Limit words that will have hypernym embeddings')
cmd:text()
opt = cmd:parse(arg or {})
table.print(opt)

i2w = torch.load(path.join(opt.dataDir, opt.i2w))
w2i = torch.load(path.join(opt.dataDir, opt.w2i))

if opt.limitVocab ~= '' then
  limit_v = {}
  for line in io.lines(path.join(opt.dataDir, opt.limitVocab)) do
    local wd = stringx.strip(line)
    limit_v[w2i[wd]] = true
  end
end

print('Loading Word2Vec embeddings...')
w2v = torch.load(opt.w2v)
num_vocab = #i2w
emb_size = w2v.M:size(2)
hemb = torch.Tensor(num_vocab, emb_size):zero()
print(string.format('Output embedding size: %dx%d', num_vocab, emb_size))

print('Processing...')
for line in io.lines(path.join(opt.dataDir, opt.hypernymFile)) do
  local parts = stringx.split(stringx.strip(line), '\t')
  local wd = parts[1]
  local w_idx = w2i[wd]
  if limit_v and limit_v[w_idx] then
    local count = 0
    for i = 2,#parts,2 do
      local c = parts[i]
      local c_idx = w2v.w2vvocab[c]
      if c_idx then
        local f = tonumber(parts[i+1])
        hemb[w_idx] = hemb[w_idx] + w2v.M[c_idx]:clone():double():mul(f)
        count = count + 1
      end
      if count >= opt.limit then
        break
      end
    end
    if count > 0 then
      hemb[w_idx] = hemb[w_idx] / hemb[w_idx]:norm()
    end
  end
end
hemb = hemb:contiguous()
print('Saving data...')
torch.save(path.join(opt.dataDir, opt.embT7), hemb)
