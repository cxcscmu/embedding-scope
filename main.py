from tqdm import tqdm
from source.embedding.miniCPM import MiniCPM
from source.dataset.textRetrieval.msMarco import MsMarco


# passageEmbeddingLoader = MsMarco.newPassageEmbeddingLoader(MiniCPM, 2048, False, 4)
# for batch in tqdm(passageEmbeddingLoader, ncols=80, ascii=False):
#     pass

# queryLoader = MsMarco.newQueryLoader("train", 2, False, 4)
# for batch in tqdm(queryLoader, ncols=80, ascii=False):
#     print(batch)
