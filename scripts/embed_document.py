import logging
import gzip
import ujson as json
import argparse
from utils import Location
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


logger = logging.getLogger("embed_document")
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    
    with open(args.input, "rt") as ifd, gzip.open(args.output, "wt") as ofd:

        texts = []
        locs = []
        for i, line in enumerate(ifd):
            item = json.loads(line)
            if item["text"]== None:
                continue
            locs.append(Location(item["location"]))
            texts.append(item["text"])
            # SANITY!
            if i > 1_000:
                logger.info(f"Read {i} lines, stopping early for sanity check.")
                break
        
        encs = model.encode(texts, 
                            batch_size=args.batch_size, 
                            show_progress_bar=True, 
                            convert_to_numpy=True, 
                            normalize_embeddings=True)
        assert len(texts) == encs.shape[0]
        
        buffer = []
        for loc, emb in tqdm(zip(locs, encs), total=len(locs), desc="Saving embeddings"):
            buffer.append(json.dumps({"location": loc, "embedding": emb.tolist()}))
            if len(buffer) >= args.batch_size:
                ofd.write("\n".join(buffer) + "\n")
                buffer = []
        if buffer:
            ofd.write("\n".join(buffer) + "\n")

            
    