/**
* Hub Miner: a hubness-aware machine learning experimentation library.
* Copyright (C) 2014  Nenad Tomasev. Email: nenad.tomasev at gmail.com
* 
* This program is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
* 
* This program is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with
* this program. If not, see <http://www.gnu.org/licenses/>.
*/
package data.neighbors.approximate;

import data.representation.DataSet;
import data.representation.DataInstance;
import data.neighbors.NeighborSetFinder;
import distances.primary.CombinedMetric;
import learning.unsupervised.Cluster;

import java.util.ArrayList;
import java.util.Random;
import java.util.Arrays;
import java.util.HashMap;

/**
 * This class implements a divide and conquer approach to approximate nearest
 * neighbor calculation via recursive Lanczos bisections. After the conquer step
 * , additional refinement of the approximate neighbor lists is performed to
 * improve the quality of the approximate kNN sets. This algorithm calculates
 * the entire approximate kNN graph. Linear algebra behind the implementation is
 * quite complex, so a read of the following paper is recommended in order to
 * better understand the code: Jie Chen, Haw-ren Fang, and Yousef Saad. 2009.
 * Fast Approximate kNN Graph Construction for High Dimensional Data via
 * Recursive Lanczos Bisection. J. Mach. Learn. Res. 10 (December 2009),
 * 1989-2012.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class AppKNNGraphLanczosBisection {

    // The point is to minimize the number of distance calculations, so we
    // won't be using the entire distance matrix.
    float[][] symDistMatrix;
    boolean[][] symDistCalcMatrix;
    DataSet ds;
    int dim;
    // Dimensionality of the Krylov subspace that we will consider.
    int s = 5;
    float alpha;
    // Neighborhood size.
    int k = 5;
    // Minimum size of a node in the data dataTree. Small nodes mean large trees
    // and more division calculations.
    int divisionThreshold = 250;
    // Metrics object.
    CombinedMetric cmet;
    // Data dataTree.
    DivDataNode dataTree;
    // Distances to kNN-s.
    float[][] kdistances;
    // An array containing the k nearest neighbors produced by the algorithm.
    int[][] kneighbors;
    float[][] tridiagonal = null;

    /**
     * @return String containing the paper reference describing the algorithm.
     */
    public static String citing() {
        return "Jie Chen, Haw-ren Fang, and Yousef Saad. 2009. Fast Approximate"
                + " kNN Graph Construction for High Dimensional Data via "
                + "Recursive Lanczos Bisection. J. Mach. Learn. Res. 10 "
                + "(December 2009), 1989-2012.";
    }

    /**
     * @param ds Data set.
     * @param symDistMatrix Upper diagonal distance matrix, if some distances
     * are known.
     * @param k Neighborhood size.
     * @param alpha Quality parameter.
     */
    public AppKNNGraphLanczosBisection(DataSet ds, float[][] symDistMatrix,
            int k, float alpha) {
        this.ds = ds;
        this.alpha = alpha;
        this.k = k;
        divisionThreshold = Math.max(5 * k, 100);
        this.symDistMatrix = symDistMatrix;
        symDistCalcMatrix = new boolean[symDistMatrix.length][];
        for (int i = 0; i < symDistMatrix.length; i++) {
            symDistCalcMatrix[i] = new boolean[symDistMatrix[i].length];
            for (int j = 0; j < symDistMatrix[i].length; j++) {
                symDistCalcMatrix[i][j] = true;
            }
        }
    }

    /**
     * @param ds Data set.
     * @param cmet Metrics object.
     * @param k Neighborhood size.
     * @param alpha Quality parameter.
     */
    public AppKNNGraphLanczosBisection(DataSet ds, CombinedMetric cmet, int k,
            float alpha) {
        this.ds = ds;
        this.alpha = alpha;
        this.k = k;
        divisionThreshold = Math.max(5 * k, 100);
        this.cmet = cmet;
    }

    /**
     * This method calculates the approximate neighbor sets by a divide and
     * conquer method based on recursive Lanczos bisections.
     */
    public void calculateApproximateNeighborSets() {
        Cluster rootCluster = new Cluster(ds);
        for (int i = 0; i < ds.size(); i++) {
            rootCluster.addInstance(i);
        }
        dataTree = new DivDataNode(rootCluster);
        dataTree.divideTree();
        KNeighborObject result = dataTree.conquerAndRefineTree();
        kdistances = result.kDists;
        kneighbors = result.kneighbors;
        System.out.print("-");
    }

    class KNeighborObject {

        float[][] kDists;
        int[][] kneighbors;
    }

    class DivDataNode {

        Cluster subset;
        DataInstance c = null;
        double[][] krylovBase; //s*n (orthogonal basis elements in the rows)
        // Symmetric s x s tridiagonal matrix.
        double[][] Tmatrix;
        double tEigenVal;
        double[] tEigenVect;
        double[] v; // The right eigenvector
        double vThreshold;
        HashMap<Integer, Integer> rightHash, leftHash, middleHash;
        DivDataNode left, right, middle; // Middle glues the left and right one.
        DivDataNode parent = null;

        public DivDataNode(Cluster subset) {
            this.subset = subset;
        }

        /**
         *
         * @return
         */
        public KNeighborObject conquerAndRefineTree() {
            if (subset.size() > divisionThreshold) {
                // In this branch, we don't calculate kNN sets exactly, but
                // rather approximately from the subtrees.
                HashMap currHash;
                // Merging.
                KNeighborObject leftKNeighbors = left.conquerAndRefineTree();
                KNeighborObject rightKNeighbors = right.conquerAndRefineTree();
                KNeighborObject middleKNeighbors =
                        middle.conquerAndRefineTree();
                KNeighborObject returnVal = new KNeighborObject();
                returnVal.kDists = new float[subset.size()][k];
                returnVal.kneighbors = new int[subset.size()][k];
                // First just merge the two lists together, then refine later.
                int rIndex = -1;
                int lIndex = -1;
                int mIndex = -1;
                int l;
                int[] kcurrLen = new int[subset.size()];
                for (int i = 0; i < subset.size(); i++) {
                    if (v[i] >= 0) {
                        ++rIndex;
                        currHash = new HashMap(4 * k);
                        for (int j = 0; j < k; j++) {
                            returnVal.kneighbors[i][j] =
                                    rightHash.get(rightKNeighbors.kneighbors[
                                    rIndex][j]);
                            returnVal.kDists[i][j] = rightKNeighbors.kDists[
                                    rIndex][j];
                            if (returnVal.kDists[i][j] < Float.MAX_VALUE) {
                                currHash.put(returnVal.kneighbors[i][j], j);
                                kcurrLen[i]++;
                            }
                        }
                    } else {
                        ++lIndex;
                        currHash = new HashMap(4 * k);
                        for (int j = 0; j < k; j++) {
                            returnVal.kneighbors[i][j] = leftHash.get(
                                    leftKNeighbors.kneighbors[lIndex][j]);
                            returnVal.kDists[i][j] = leftKNeighbors.kDists[
                                    lIndex][j];
                            if (returnVal.kDists[i][j] < Float.MAX_VALUE) {
                                currHash.put(returnVal.kneighbors[i][j], j);
                                kcurrLen[i]++;
                            }
                        }
                    }
                    if (Math.abs(v[i]) <= vThreshold) {
                        mIndex++;
                        // Now insert the overlapping data.
                        for (int j = 0; j < k; j++) {
                            if (currHash.containsKey(middleHash.get(
                                    middleKNeighbors.kneighbors[mIndex][j]))) {
                                continue;
                            }
                            if (kcurrLen[i] > 0) {
                                if (kcurrLen[i] == k) {
                                    if (middleKNeighbors.kDists[mIndex][j]
                                            < returnVal.kDists[i][
                                            kcurrLen[i] - 1]) {
                                        // Search to see where to insert.
                                        l = k - 1;
                                        while ((l >= 1)
                                                && middleKNeighbors.kDists[
                                                mIndex][j] < returnVal.
                                                kDists[i][l - 1]) {
                                            returnVal.kDists[i][l] =
                                                    returnVal.kDists[i][l - 1];
                                            returnVal.kneighbors[i][l] =
                                                    returnVal.kneighbors[i][
                                                    l - 1];
                                            l--;
                                        }
                                        returnVal.kDists[i][l] =
                                                middleKNeighbors.kDists[
                                                mIndex][j];
                                        returnVal.kneighbors[i][l] =
                                                middleHash.get(
                                                middleKNeighbors.kneighbors[
                                                mIndex][j]);
                                    }
                                } else {
                                    if (middleKNeighbors.kDists[mIndex][j]
                                            < returnVal.kDists[i][
                                            kcurrLen[i] - 1]) {
                                        // Search to see where to insert.
                                        l = kcurrLen[i] - 1;
                                        returnVal.kDists[i][kcurrLen[i]] =
                                                returnVal.kDists[i][
                                                kcurrLen[i] - 1];
                                        returnVal.kneighbors[i][kcurrLen[i]] =
                                                returnVal.kneighbors[i][
                                                kcurrLen[i] - 1];
                                        while ((l >= 1)
                                                && middleKNeighbors.
                                                kDists[mIndex][
                                                j] < returnVal.kDists[i][
                                                l - 1]) {
                                            returnVal.kDists[i][l] =
                                                    returnVal.kDists[i][l - 1];
                                            returnVal.kneighbors[i][l] =
                                                    returnVal.kneighbors[i][
                                                    l - 1];
                                            l--;
                                        }
                                        returnVal.kDists[i][l] =
                                                middleKNeighbors.kDists[
                                                mIndex][j];
                                        returnVal.kneighbors[i][l] =
                                                middleHash.get(
                                                middleKNeighbors.kneighbors[
                                                mIndex][j]);
                                        kcurrLen[i]++;
                                    } else {
                                        returnVal.kDists[i][kcurrLen[i]] =
                                                middleKNeighbors.kDists[
                                                mIndex][j];
                                        returnVal.kneighbors[i][kcurrLen[i]] =
                                                middleHash.get(
                                                middleKNeighbors.kneighbors[
                                                mIndex][j]);
                                        kcurrLen[i]++;
                                    }
                                }
                            } else {
                                returnVal.kDists[i][0] =
                                        middleKNeighbors.kDists[mIndex][j];
                                returnVal.kneighbors[i][0] =
                                        middleHash.get(
                                        middleKNeighbors.kneighbors[mIndex][j]);
                                kcurrLen[i] = 1;
                            }
                        }
                    }
                }
                // k neighbors are refined by also taking into account the 
                // neighbors of neighbors
                ArrayList<Integer> candidates;
                ArrayList<Float> candidateDists;
                int iIndex, jIndex;
                int minInd, maxInd;
                for (int i = 0; i < subset.size(); i++) {
                    iIndex = subset.getIndexes().get(i);
                    // currHas holds the "forbidden" elements (to avoid
                    // repetitions).
                    currHash = new HashMap(k * k);
                    candidates = new ArrayList<>(k * k);
                    candidateDists = new ArrayList<>(k * k);
                    for (int j = 0; j < k; j++) {
                        currHash.put(i, -1);
                        if (returnVal.kDists[i][j] < Float.MAX_VALUE) {
                            currHash.put(returnVal.kneighbors[i][j], j);
                        }
                    }
                    for (int j = 0; j < k; j++) {
                        if (returnVal.kDists[i][j] < Float.MAX_VALUE) {
                            for (int j1 = 0; j1 < k; j1++) {
                                if (returnVal.kDists[
                                        returnVal.kneighbors[i][j]][j1]
                                        < Float.MAX_VALUE) {
                                    if (!currHash.containsKey(
                                            returnVal.kneighbors[
                                            returnVal.kneighbors[i][j]][j1])) {
                                        candidates.add(
                                                returnVal.kneighbors[
                                                returnVal.kneighbors[i][j]][
                                                j1]);
                                        jIndex = subset.getIndexes().get(
                                                returnVal.kneighbors[
                                                returnVal.kneighbors[i][j]][
                                                j1]);
                                        minInd = Math.min(iIndex, jIndex);
                                        maxInd = Math.max(iIndex, jIndex);
                                        if (!symDistCalcMatrix[minInd][
                                                maxInd - minInd - 1]) {
                                            try {
                                                symDistMatrix[minInd][
                                                        maxInd - minInd - 1] =
                                                        cmet.dist(subset.
                                                        getInstance(i),
                                                        subset.getInstance(
                                                        returnVal.kneighbors[
                                                        returnVal.kneighbors[
                                                        i][j]][j1]));
                                            } catch (Exception e) {
                                                System.err.println(
                                                        e.getMessage());
                                            }
                                            symDistCalcMatrix[minInd][
                                                    maxInd - minInd - 1] = true;
                                        }
                                        candidateDists.add(
                                                symDistMatrix[minInd][
                                                maxInd - minInd - 1]);
                                        currHash.put(
                                                returnVal.kneighbors[
                                                    returnVal.kneighbors[i][j]][
                                                j1], k * j + j1);
                                    }
                                }
                            }
                        }
                    }
                    for (int j = 0; j < candidates.size(); j++) {
                        // Now refine the k neighbor set by checking out the
                        // candidates.
                        if (kcurrLen[i] > 0) {
                            if (kcurrLen[i] == k) {
                                if (candidateDists.get(j)
                                        < returnVal.kDists[i][
                                        kcurrLen[i] - 1]) {
                                    // Search to see where to insert.
                                    l = k - 1;
                                    while ((l >= 1)
                                            && candidateDists.get(j)
                                            < returnVal.kDists[i][l - 1]) {
                                        returnVal.kDists[i][l] =
                                                returnVal.kDists[i][l - 1];
                                        returnVal.kneighbors[i][l] =
                                                returnVal.kneighbors[i][l - 1];
                                        l--;
                                    }
                                    returnVal.kDists[i][l] =
                                            candidateDists.get(j);
                                    returnVal.kneighbors[i][l] =
                                            candidates.get(j);
                                }
                            } else {
                                if (candidateDists.get(j)
                                        < returnVal.kDists[i][
                                        kcurrLen[i] - 1]) {
                                    // Search to see where to insert.
                                    l = kcurrLen[i] - 1;
                                    returnVal.kDists[i][kcurrLen[i]] =
                                            returnVal.kDists[i][
                                            kcurrLen[i] - 1];
                                    returnVal.kneighbors[i][kcurrLen[i]] =
                                            returnVal.kneighbors[i][
                                            kcurrLen[i] - 1];
                                    while ((l >= 1)
                                            && candidateDists.get(j)
                                            < returnVal.kDists[i][l - 1]) {
                                        returnVal.kDists[i][l] =
                                                returnVal.kDists[i][l - 1];
                                        returnVal.kneighbors[i][l] =
                                                returnVal.kneighbors[i][l - 1];
                                        l--;
                                    }
                                    returnVal.kDists[i][l] =
                                            candidateDists.get(j);
                                    returnVal.kneighbors[i][l] =
                                            candidates.get(j);
                                    kcurrLen[i]++;
                                } else {
                                    returnVal.kDists[i][kcurrLen[i]] =
                                            candidateDists.get(j);
                                    returnVal.kneighbors[i][kcurrLen[i]] =
                                            candidates.get(j);
                                    kcurrLen[i]++;
                                }
                            }
                        } else {
                            returnVal.kDists[i][0] = candidateDists.get(j);
                            returnVal.kneighbors[i][0] = candidates.get(j);
                            kcurrLen[i] = 1;
                        }


                    }
                }
                return returnVal;

            } else {
                KNeighborObject returnVal = new KNeighborObject();
                if (subset.isEmpty()) {
                    return returnVal;
                }
                DataSet dcNSF = subset.getDefinitionDataset().cloneDefinition();
                dcNSF.data = subset.getAllInstances();
                // Now calculate the distance matrix for the neighbor set finder
                if (symDistMatrix == null) {
                    symDistMatrix = new float[ds.size()][];
                    symDistCalcMatrix = new boolean[ds.size()][];
                    for (int i = 0; i < ds.size(); i++) {
                        symDistMatrix[i] = new float[ds.size() - i - 1];
                        symDistCalcMatrix[i] = new boolean[ds.size() - i - 1];
                    }
                }
                float[][] distMat = new float[subset.size()][];
                int minInd, maxInd;
                for (int i = 0; i < subset.size(); i++) {
                    distMat[i] = new float[subset.size() - i - 1];
                    for (int j = i + 1; j < subset.size(); j++) {
                        minInd = Math.min(subset.getIndexes().get(i),
                                subset.getIndexes().get(j));
                        maxInd = Math.max(subset.getIndexes().get(i),
                                subset.getIndexes().get(j));
                        if (!symDistCalcMatrix[minInd][maxInd - minInd - 1]) {
                            try {
                                symDistMatrix[minInd][maxInd - minInd - 1] =
                                        cmet.dist(subset.getInstance(i),
                                        subset.getInstance(j));
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                            symDistCalcMatrix[minInd][maxInd - minInd - 1] =
                                    true;
                        }
                        distMat[i][j - i - 1] =
                                symDistMatrix[minInd][maxInd - minInd - 1];
                    }
                }
                NeighborSetFinder nsf =
                        new NeighborSetFinder(dcNSF, distMat, cmet);
                nsf.calculateNeighborSets(k);
                returnVal.kneighbors = nsf.getKNeighbors();
                returnVal.kDists = nsf.getKDistances();
                return returnVal;
            }
        }

        /**
         * This method splits a node in the dataTree.
         */
        public void divideTree() {
            if (subset.size() > divisionThreshold) {
                doAllPreSplitNodeCalculations();
                double[] vAbs = new double[v.length];
                for (int i = 0; i < subset.size(); i++) {
                    vAbs[i] = Math.abs(v[i]);
                }
                Cluster leftCluster = new Cluster(
                        subset.getDefinitionDataset(), subset.size() * 2 / 3);
                Cluster rightCluster = new Cluster(
                        subset.getDefinitionDataset(), subset.size() * 2 / 3);
                Cluster middleCluster = new Cluster(
                        subset.getDefinitionDataset(),
                        (int) (subset.size() * alpha) + 10);
                // Ascending sort.
                Arrays.sort(vAbs);
                vThreshold = vAbs[(int) (alpha * subset.size())];
                // HashMaps keep track of which element was assigned to which
                // subtree.
                rightHash = new HashMap(subset.size());
                leftHash = new HashMap(subset.size());
                middleHash = new HashMap(subset.size());
                for (int i = 0; i < subset.size(); i++) {
                    if (v[i] >= 0) {
                        rightHash.put(rightCluster.size(), i);
                        rightCluster.addInstance(
                                subset.getWithinDataSetIndexOf(i));
                    } else {
                        leftHash.put(leftCluster.size(), i);
                        leftCluster.addInstance(
                                subset.getWithinDataSetIndexOf(i));
                    }
                    if (Math.abs(v[i]) <= vThreshold) {
                        middleHash.put(middleCluster.size(), i);
                        // Alpha perc of the data in the node goes to the middle
                        // subnode, glueing the left and the right ones.
                        middleCluster.addInstance(
                                subset.getWithinDataSetIndexOf(i));
                    }
                }
                left = new DivDataNode(leftCluster);
                right = new DivDataNode(rightCluster);
                middle = new DivDataNode(middleCluster);
                left.parent = this;
                right.parent = this;
                middle.parent = this;
                left.divideTree();
                right.divideTree();
                middle.divideTree();
                // Here we reconstruct the original feature values.
                uncenterCollection();
            }
        }

        public void doAllPreSplitNodeCalculations() {
            dim = ds.getNumFloatAttr();
            // Uncentering is done on a subtree after it has been constructed,
            // previous to finding the approximate nearest neighbors.
            centerCollection();
            calculateKrylovBase();
            calculateRightEigenvector();
        }

        /**
         * This method calculates the right eigenvector. It is reconstructed
         * based on the right eigenvector of the symmetric tridiagonal matrix
         * obtained from multiplying regular and transposed Krylov base matrices
         * and data matrices.
         */
        public void calculateRightEigenvector() {
            // Here we have a very small matrix, so using a power method is ok.
            tEigenVect = new double[s];
            for (int i = 0; i < s; i++) {
                tEigenVect[i] = 1 / (float) Math.sqrt(s);
            }
            double[] tEigenNew;
            boolean convergence = false;
            float error;
            double normFact;
            int iter = 0;
            while (!convergence) {
                iter++;
                error = 0;
                normFact = 0;
                tEigenNew = new double[s];
                // Calculate the new eigenvector.
                for (int i = 0; i < s; i++) {
                    for (int j = Math.max(0, i - 1);
                            j < Math.min(s - 1, i + 1); j++) {
                        tEigenNew[i] += Tmatrix[i][j] * tEigenVect[j];
                    }
                    normFact += tEigenNew[i] * tEigenNew[i];
                }
                normFact = Math.sqrt(normFact);
                for (int i = 0; i < s; i++) {
                    tEigenNew[i] /= normFact;
                }
                tEigenVal = normFact;
                for (int i = 0; i < s; i++) {
                    error += Math.abs(tEigenNew[i] - tEigenVect[i]);
                }
                for (int i = 0; i < s; i++) {
                    tEigenVect[i] = tEigenNew[i];
                }
                if (error < s * 0.05 || iter > 20) {
                    convergence = true;
                }
            }
            // So, now we have an eigenvector of T matrix... (tridiagonal matrix
            // reached by multiplying with the transformation matrix from both
            // sides) Now it's easy to calculate the needed right eigenvector.
            v = new double[subset.size()];
            for (int i = 0; i < v.length; i++) {
                for (int j = 0; j < s; j++) {
                    v[i] += krylovBase[j][i] * tEigenVect[j];
                }
            }
        }

        /**
         * This method calculates the base of the Krylov subspace.
         */
        public void calculateKrylovBase() {
            //Golub-Kahan-Lanczos bidiagonalization
            Random randa = new Random();
            // vTemp is an arbitrary vector.
            float[] vTemp = new float[subset.size()];
            float tSum = 0;
            for (int i = 0; i < vTemp.length; i++) {
                vTemp[i] = (float) randa.nextGaussian();
                tSum += vTemp[i] * vTemp[i];
            }
            // Normalize.
            tSum = (float) Math.sqrt(tSum);
            for (int i = 0; i < vTemp.length; i++) {
                vTemp[i] /= tSum;
            }
            // Calculate the vectors defining the span.
            double[][] vVectors = new double[s + 1][subset.size()];
            for (int i = 0; i < vTemp.length; i++) {
                vVectors[1][i] = vTemp[i];
            }

            double[][] uVectors = new double[s + 1][dim];
            double[] alpha = new double[s + 1];
            double[] beta = new double[s + 1];

            // Here is the iterative Lanczos procedure.
            for (int i = 1; i <= s - 1; i++) {
                for (int j = 0; j < dim; j++) {
                    for (int t = 0; t < subset.size(); t++) {
                        uVectors[i][j] += vVectors[i][t]
                                * subset.getInstance(t).fAttr[j];
                    }
                    if (i > 1) {
                        uVectors[i][j] -= beta[i - 1] * uVectors[i - 1][j];
                    }
                    alpha[i] += uVectors[i][j] * uVectors[i][j];
                }
                alpha[i] = Math.sqrt(alpha[i]);
                for (int j = 0; j < dim; j++) {
                    uVectors[i][j] /= alpha[i];
                }
                for (int t = 0; t < subset.size(); t++) {
                    for (int j = 0; j < dim; j++) {
                        vVectors[i + 1][t] += uVectors[i][j]
                                * subset.getInstance(t).fAttr[j];
                    }
                    vVectors[i + 1][t] -= alpha[i] * vVectors[i][t];
                    beta[i] += vVectors[i + 1][t] * vVectors[i + 1][t];
                }
                beta[i] = Math.sqrt(beta[i]);
                for (int t = 0; t < subset.size(); t++) {
                    vVectors[i + 1][t] /= beta[i];
                }
            }
            // Now calculate the last alpha.
            for (int j = 0; j < dim; j++) {
                for (int t = 0; t < subset.size(); t++) {
                    uVectors[s][j] += vVectors[s][t]
                            * subset.getInstance(t).fAttr[j];
                }
                uVectors[s][j] -= beta[s - 1] * uVectors[s - 1][j];
                alpha[s] += uVectors[s][j] * uVectors[s][j];
            }
            alpha[s] = Math.sqrt(alpha[s]);

            // It is derived from the bidiagonal form.
            Tmatrix = new double[s][s];
            Tmatrix[0][0] = alpha[1] * alpha[1];
            Tmatrix[0][1] = alpha[1] * beta[1];
            Tmatrix[s - 1][s - 1] = alpha[s] * alpha[s] + beta[s - 1] *
                    beta[s - 1];
            Tmatrix[s - 1][s - 2] = alpha[s - 1] * beta[s - 1];
            for (int i = 1; i < s - 1; i++) {
                Tmatrix[i][i] = alpha[i + 1] * alpha[i + 1] + beta[i] *
                        beta[i];
                Tmatrix[i][i - 1] = alpha[i] * beta[i];
                Tmatrix[i][i + 1] = Tmatrix[i][i - 1];
            }
            krylovBase = new double[s][subset.size()];
            for (int i = 1; i < s + 1; i++) {
                krylovBase[i - 1] = vVectors[i];
            }
            System.gc();
        }

        /**
         * Subtract the centroid feature values from the instance feature
         * vectors.
         */
        public void centerCollection() {
            int dim = subset.getDefinitionDataset().getNumFloatAttr();
            try {
                c = subset.getCentroid();
            } catch (Exception e) {
                e.printStackTrace();
            }
            for (DataInstance dInst : subset.getAllInstances()) {
                for (int d = 0; d < dim; d++) {
                    dInst.fAttr[d] -= c.fAttr[d];
                }
            }
        }

        /**
         * Add centroid feature values to instance feature vectors.
         */
        public void uncenterCollection() {
            int dim = subset.getDefinitionDataset().getNumFloatAttr();
            for (DataInstance dInst : subset.getAllInstances()) {
                for (int d = 0; d < dim; d++) {
                    dInst.fAttr[d] += c.fAttr[d];
                }
            }
        }
    }

    /**
     * @return DataSet data.
     */
    public DataSet getDataSet() {
        return ds;
    }

    /**
     * @return CombinedMetric metrics object.
     */
    public CombinedMetric getMetric() {
        return cmet;
    }

    /**
     * @return Upper diagonal distance matrix (many distances might not be
     * calculated, as this class avoids that).
     */
    public float[][] getDistances() {
        return symDistMatrix;
    }

    /**
     * @return A boolean array indicating which distances have been calculated.
     */
    public boolean[][] getDistanceFlags() {
        return symDistCalcMatrix;
    }

    /**
     * @return Neighborhood size.
     */
    public int getK() {
        return k;
    }

    /**
     * @return kNN sets, given as a two-dimensional array.
     */
    public int[][] getKneighbors() {
        return kneighbors;
    }

    /**
     * @return kNN distances, given as a two-dimensional array.
     */
    public float[][] getKdistances() {
        return kdistances;
    }
}
