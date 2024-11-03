from source.dataset.msMarco import MsMarco
from source.embedding.miniCPM import MiniCPM

dataset = MsMarco()
passageID, passageEmbedding = next(dataset.getPassageEmbeddings(MiniCPM))
print(passageID, type(passageID))
print(passageEmbedding)
