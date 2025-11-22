import argparse
import os
import numpy as np
import csv
from PIL import Image
from multiprocessing import Pool

class EvalHierarchiness:
    def __init__(self, *partitions):
        self.npartitions = len(partitions)
        self.partitions = partitions
        self.nrows = [np.max(p) for p in partitions]
        self.ncols = [np.max(p) for p in partitions]
        self.mxy = []

        for i in range(self.npartitions - 1):
            m = self.compute_cooccurrence_matrix(self.partitions[i], self.partitions[i + 1])
            self.mxy.append(m)

        self.nelems = [np.sum(m) for m in self.mxy]
        self.size_x = [np.sum(m, axis=1) for m in self.mxy]
        self.size_y = [np.sum(m, axis=0) for m in self.mxy]
        
        self.mxy_x = [np.divide(m, sx[:, None], where=sx[:, None] != 0) for m, sx in zip(self.mxy, self.size_x)]
        self.mxy_y = [np.divide(m, sy, where=sy != 0) for m, sy in zip(self.mxy, self.size_y)]
    #enddef

    def compute_cooccurrence_matrix(self, p1, p2):
        max_r, max_c = np.max(p1), np.max(p2)

        m = np.zeros((max_r + 1, max_c + 1), dtype=int)
        np.add.at(m, (p1, p2), 1)
        
        return m[1:, 1:]

    def _compute_norm_factor(self, norm):
        if norm == "numreg":
            return np.ones_like(self.size_y[0]) / self.size_y[0].shape[0]
        elif norm == "global":
            return self.size_y[0] / self.nelems[0]
        elif norm == "regsize":
            return self.size_y[0]
        else:
            return np.ones_like(self.size_y[0])
        #endif
    #enddef

    def _compute_cover_nucl(self):
        count_cover = np.sum(self.mxy_y > 0, axis=0)
        count_nucl = np.sum(self.mxy_x == 1, axis=0)
        return count_cover, count_nucl
    #enddef

    def _compute_norm_factor_dif(self, norm):
        norm_factors = []
        for sy, ne in zip(self.size_y, self.nelems):
            if norm == "global":
                factor = np.ones_like(sy, dtype=float) / ne if ne != 0 else np.ones_like(sy, dtype=float)
            elif norm == "regional":
                factor = np.divide(1.0, sy, out=np.ones_like(sy, dtype=float), where=sy != 0)
            else:
                factor = np.ones_like(sy, dtype=float)
            norm_factors.append(factor)
        return norm_factors
    # enddef

    def eval_infl_pairwise(self, norm="none"):
        norm_factors = self._compute_norm_factor_dif(norm)
        infl_results = []
        
        for mxy, mxy_x, mxy_y, norm_factor in zip(self.mxy, self.mxy_x, self.mxy_y, norm_factors):
            tmp = np.logical_and(mxy_x < 1, mxy_y < 1)
            res = np.sum(tmp * mxy * norm_factor)
            infl_results.append(res)
        
        return infl_results
    #enddef

    def eval_infl_recursive(self, norm="none", idx=0):
        norm_factors = self._compute_norm_factor_dif(norm)
        n_pairs = len(self.mxy_x)

        def rec(i):
            if i >= n_pairs:
                return 0.0
            mxy_x = self.mxy_x[i]
            mxy_y = self.mxy_y[i]
            m = self.mxy[i]
            nf = norm_factors[i]
            tmp = np.logical_and(mxy_x < 1, mxy_y < 1).astype(float)
            val = np.sum(tmp * m * nf)
            return val + rec(i + 1)

        total = rec(0)
        return total / n_pairs if n_pairs > 0 else 0.0
    # enddef

    def prom_part(self, infl_list):
        scores = []
        for i in range(len(infl_list) + 1):
            if i == 0:
                score = infl_list[0]
            elif i == len(infl_list):
                score = infl_list[-1]
            else:
                score = (infl_list[i - 1] + infl_list[i]) / 2
            scores.append(score)

        ranking = np.argsort(scores)[::-1]
        return ranking, np.sort(scores)[::-1]
    # enddef

    def optimize_partitions(self, ranking, norm="none"):
        optimized_partitions = list(self.partitions)
        original_infl = self.eval_infl_recursive(norm)
        
        for idx in ranking:
            temp_partitions = optimized_partitions[:idx] + optimized_partitions[idx + 1:]
            temp_eval = EvalHierarchiness(*temp_partitions)
            new_infl = temp_eval.eval_infl_recursive(norm)
            
            if new_infl < original_infl:
                optimized_partitions = temp_partitions
                original_infl = new_infl
                print(f"Removed partition {idx}, new inflation: {new_infl}")
                break
            else:
                print(f"Partition {idx} removal did not reduce inflation.")

        return optimized_partitions
    
def main(args):
    # Partition A
    pa = np.array([[1,1,2,2,5,5,5,5],
                  [1,1,2,2,5,5,5,5],
                  [3,3,4,4,5,5,5,5],
                  [3,3,4,4,5,5,5,5],
                  [6,6,6,6,7,7,8,8],
                  [6,6,6,6,7,7,8,8],
                  [6,6,6,6,9,9,10,10],
                  [6,6,6,6,9,9,10,10],
                 ]);
     # Partition B
    pb = np.array([[1,1,1,2,5,5,6,6],
                  [1,1,1,2,5,5,6,6],
                  [1,1,1,2,5,5,6,6],
                  [3,3,3,4,8,8,8,8],
                  [7,7,7,7,8,8,8,8],
                  [7,7,7,7,8,8,8,8],
                  [7,7,7,7,8,8,8,8],
                  [7,7,7,7,9,9,10,10],
                 ]);
     # Partition C
    pc = np.array([[1,1,1,1,2,2,2,2],
                  [1,1,1,1,2,2,2,2],
                  [1,1,1,1,2,2,2,2],
                  [1,1,1,1,2,2,2,2],
                  [3,3,3,3,4,4,4,4],
                  [3,3,3,3,4,4,4,4],
                  [3,3,3,3,4,4,4,4],
                  [3,3,3,3,4,4,4,4],
                 ]);

    partitions = [np.array(Image.open(p)) for p in args.partitions] if args.partitions else [pa, pb, pc]
    
    eval_part = EvalHierarchiness(*partitions)

    print(50*"-" + "\nHierarchiness Measures\n" + 50*"-")
    if args.norm in ["global", "none", "regional"]:
        infl = eval_part.eval_infl_pairwise(args.norm)
        ranking, scores = eval_part.prom_part(infl)
        infl = list(map(float, infl))
        
        print("Inflation Ratio(X,Y):", infl)
        print("Recursive Inflation Ratio(X,Y):", eval_part.eval_infl_recursive(args.norm))
        print("Most problematic partitions:", ranking, "with scores:", scores)
        optimized_partitions = eval_part.optimize_partitions(ranking, args.norm)
    else:
        infl = eval_part.eval_infl_pairwise("global")
        ranking, scores = eval_part.prom_part(infl)
        infl = list(map(float, infl))
        
        print("Inflation Ratio(X,Y):", infl)
        print("Recursive Inflation Ratio(X,Y):", eval_part.eval_infl_recursive("global"))
        print("Most problematic partitions:", ranking, "with scores:", scores)
        optimized_partitions = eval_part.optimize_partitions(ranking, "global")
#end main

if __name__ == '__main__':
    parser = argparse.ArgumentParser();
    parser.add_argument("--partitions", nargs="+", help="Path to the label image of each partition", type=str);
    parser.add_argument("--norm", help="Normalization factor: none, regional, global", type=str);

    args = parser.parse_args();
    main(args);
#endif