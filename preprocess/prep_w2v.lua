require 'pl'
require 'dp'
require 'torch'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Subselect embeddings a word2vec torch binary file.')
cmd:option('--w2vPath', '/websail/common/embeddings/word2vec/GoogleNews-vectors-negative300.t7', 'Torch binary file of Mikolov word2vec, see word2vec.torch for conversion.')
cmd:option('--dataDir', 'data/commondefs', 'Data directory')
cmd:option('--word2IndexT7', 'word2index.t7',
           'Input map from word to index in torch binary file')
cmd:option('--index2WordT7', 'index2word.t7',
          'Input map from index to word in torch binary file')
cmd:option('--embT7', 'emb.t7',
          'Output embeddings')
cmd:option('--unkT7', 'unk_emb_id.t7',
          'Output missing embedding word id')
cmd:option('--unkSym', '<unk>', 'Unknown symbol')
cmd:option('--startSym', '<s>', 'Start sentence symbol')
cmd:option('--endSym', '</s>', 'End sentence symbol')
cmd:option('--w2vUnkSym', 'UNK', 'Unknown symbol')
cmd:option('--w2vEndSym', '</s>', 'Start sentence symbol')
cmd:option('--unkUniform', 0.05, 'initialize embeddings that do not exist in word2vec, -1 means UNK embedding')


cmd:text()
opt = cmd:parse(arg or {})
table.print(opt)

local id2word = torch.load(path.join(opt.dataDir, opt.index2WordT7))
local num_vocab = #id2word

print('Loading w2v, this will take a minute...')
local word2vec = torch.load(opt.w2vPath)
local emb = torch.Tensor(num_vocab, word2vec.M:size(2))
local w2v_unk_id = word2vec.w2vvocab[opt.w2vUnkSym]
local unk_ids = {}

print('Copying w2v...')
for i = 1, num_vocab do
  local w = id2word[i]
  if w == opt.unkSym then w = opt.w2vUnkSym end
  if w == opt.endSym then w = opt.w2vEndSym end
  local w2v_id = word2vec.w2vvocab[w]
  if w == opt.startSym then
    emb[i]:zero()
  elseif not w2v_id then
    emb[i] = word2vec.M[w2v_unk_id]
    unk_ids[i] = true
    if opt.unkUniform > 0 then
      emb[i]:uniform(-opt.unkUniform, opt.unkUniform)
    end
  else
    emb[i] = word2vec.M[w2v_id]
  end
end
emb = emb:contiguous()
print('Saving data...')
torch.save(path.join(opt.dataDir, opt.embT7), emb)
torch.save(path.join(opt.dataDir, opt.unkT7), unk_ids)
