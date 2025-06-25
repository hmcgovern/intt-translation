import random
import logging
import gzip
import json
import re
import argparse
import numpy
from utils import Location


logger = logging.getLogger("score_embeddings")

def bootstrap_ratio(X, Y, n_bootstrap=1000, ci=95):
        # Array to store bootstrapped ratios
        bootstrapped_ratios = numpy.empty(n_bootstrap)
    
        # Perform bootstrapping
        for i in range(n_bootstrap):
            # Sample with replacement from X and Y
            X_resample = numpy.random.choice(X, size=len(X), replace=True)
            Y_resample = numpy.random.choice(Y, size=len(Y), replace=True)
        
            # Calculate the ratio for this resample
            bootstrapped_ratios[i] = numpy.mean(X_resample) / numpy.mean(Y_resample)
    
        # Calculate the confidence interval percentiles
        lower_bound = numpy.percentile(bootstrapped_ratios, (100 - ci) / 2)
        upper_bound = numpy.percentile(bootstrapped_ratios, 100 - (100 - ci) / 2)
    
        # Return the bootstrapped ratio and the confidence interval
        return numpy.mean(bootstrapped_ratios), lower_bound, upper_bound


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", dest="gold", help="Input file")
    parser.add_argument("--embeddings", nargs="+", dest="embeddings", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--vote_threshold", dest="vote_threshold", default=50, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    
    books = set()
    labels = set()
    golds = {}
    num_golds = 0
    with open(args.gold, "rt") as ifd:
        ifd.readline()
        for line in ifd:
            if "-" not in line:
                a, b, votes = line.strip().split()
                a = Location(a)
                b = Location(b)
                votes = int(votes)
                labels.add(a)
                labels.add(b)
                books.add(a["book"])
                books.add(b["book"])
                if a < b:
                    src, tgt = a, b
                else:
                    src, tgt = b, a
                key = (src, tgt)
                golds[src] = golds.get(src, {})
                if votes > args.vote_threshold:
                    golds[src][tgt] = votes
                    num_golds += 1


    embs = {}
    for emb in args.embeddings:
        with gzip.open(emb, "rt") as ifd:
            for line in ifd:
                j = json.loads(line)
                loc = Location(j["location"])
                bk = loc["book"]
                ch = loc["chapter"]
                v = loc["verse"]
                embs[bk] = embs.get(bk, {})
                embs[bk][ch] = embs[bk].get(ch, {})
                embs[bk][ch][v] = numpy.array(j["embedding"])

    ref_dists = []
    nonref_dists = []
    for src_loc, tgts in golds.items():
        src_bk = src_loc["book"]
        src_ch = src_loc["chapter"]
        src_v = src_loc["verse"]
        
        for tgt_loc, votes in tgts.items():
            tgt_bk = tgt_loc["book"]
            tgt_ch = tgt_loc["chapter"]
            tgt_v = tgt_loc["verse"]
            
            try:
                src_emb = embs[src_bk][src_ch][src_v]
                tgt_emb = embs[tgt_bk][tgt_ch][tgt_v]
            except:
                continue
            ref_dists.append(numpy.dot(src_emb, tgt_emb))
            poss = [v for k, v in embs[tgt_bk][tgt_ch].items() if k != tgt_v]
            random.shuffle(poss)
            nonref_dists.append(numpy.dot(src_emb, poss[0]))



    mean_ratio, lower_ci, upper_ci = bootstrap_ratio(ref_dists, nonref_dists, n_bootstrap=10000, ci=95)
    print(f"Mean ratio: {mean_ratio}, 95% CI: [{lower_ci}, {upper_ci}]")



    with open(args.output, "wt") as ofd:
        s_ref_dists = numpy.mean(ref_dists) 
        s_nonref_dists = numpy.mean(nonref_dists)
        ofd.write(json.dumps({"reference_similarity" : s_ref_dists,  
                              "nonreference_similarity" : s_nonref_dists, 
                              "ratio" : s_ref_dists / s_nonref_dists,
                              "CI": (lower_ci, upper_ci), 
                              "num_refs" : len(ref_dists), 
                              "file_name" : args.embeddings,})  + "\n")