local JoinOneToMany, parent = torch.class('nn.JoinOneToMany', 'nn.Container')

-- based on ZipTableOneToMany in dpnn

-- input : { v, [x1, x2, ...] }
-- output : [v:x1, v:x2, ...]
-- only join the last dimension, only support batch mode
-- TODO: handle other dimension
function JoinOneToMany:__init()
  parent.__init(self)
  self:_initState()
end

function JoinOneToMany:_initState()
  self.output = torch.Tensor(1,1,1)
  self.gradInput = {}
  self.gradInputV = torch.Tensor(1,1)
  self.gradInputX = torch.Tensor(1,1,1)
end

function JoinOneToMany:clearState()
  parent.clearState(self)
  self:_initState()
end

function JoinOneToMany:updateOutput(input)
  assert(#input == 2, "input must be table")
  local inputV, inputX = input[1], input[2]
  local vsize = inputV:size(2)
  local xsize = inputX:size(3)
  if self.output:type() ~= inputX:type() or
    self.output:size(1) ~= inputX:size(1) or
    self.output:size(2) ~= inputX:size(2) or
    self.output:size(3) ~= inputX:size(3) + inputV:size(2) then
    self.output = torch.Tensor()
      :typeAs(inputX)
      :resize(inputX:size(1),inputX:size(2),inputX:size(3) + inputV:size(2))
  end
  self.output[{{},{},{vsize+1,vsize+xsize}}]:copy(inputX)
  for i = 1, inputX:size(1) do
    self.output[{{i},{},{1,vsize}}]:copy(inputV)
  end
  return self.output
end

function JoinOneToMany:updateGradInput(input, gradOutput)
  assert(#input == 2, "input must be table")
  local inputV, inputX = input[1], input[2]
  local vsize = inputV:size(2)
  local xsize = inputX:size(3)
  if self.gradInputX:type() ~= inputX:type() or
    self.gradInputX:size(1) ~= inputX:size(1) or
    self.gradInputX:size(2) ~= inputX:size(2) or
    self.gradInputX:size(3) ~= inputX:size(3) + inputV:size(2) then
    self.gradInputX = torch.Tensor():typeAs(inputX):resizeAs(inputX)
  end
  self.gradInputX:copy(gradOutput[{{},{},{vsize+1,vsize+xsize}}])
  if self.gradInputV:type() ~= inputV:type() or
    self.gradInputV:size(1) ~= inputV:size(1) or
    self.gradInputV:size(2) ~= inputV:size(2) then
    self.gradInputV = torch.Tensor():typeAs(inputV):resizeAs(inputV)
  end
  self.gradInputV:zero()
  for i = 1, inputX:size(1) do
    self.gradInputV:add(gradOutput[{{i},{},{1,vsize}}])
  end
  self.gradInput = {self.gradInputV, self.gradInputX}
  return self.gradInput
end
