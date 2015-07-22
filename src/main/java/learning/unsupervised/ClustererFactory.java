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
package learning.unsupervised;

import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.kernel.Kernel;
import distances.kernel.KernelMatrixUserInterface;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import learning.unsupervised.methods.DBScan;
import learning.unsupervised.methods.FastKMeans;
import learning.unsupervised.methods.FastKMeansPlusPlus;
import learning.unsupervised.methods.GHPC;
import learning.unsupervised.methods.GHPKM;
import learning.unsupervised.methods.GKH;
import learning.unsupervised.methods.HarmonicKMeans;
import learning.unsupervised.methods.KMeans;
import learning.unsupervised.methods.KMeansPlusPlus;
import learning.unsupervised.methods.KMedoids;
import learning.unsupervised.methods.KMedoidsPlusPlus;
import learning.unsupervised.methods.KernelGHPKM;
import learning.unsupervised.methods.KernelKMeans;
import learning.unsupervised.methods.LHPC;
import learning.unsupervised.methods.LKH;

/**
 * This class is used to fetch the initial parametrizations of various
 * clustering method objects within the evaluation framework.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClustererFactory {
    
    /**
     * This is how initial clusterer instances are currently generated, as they
     * require initial parametrizations and doing all of that via reflection in
     * every single case might be somewhat non-trivial in the general case -
     * though it is certainly preferable to having the algorithm names hardcoded
     * inside. TODO: change this clusterer initialization mechanism.
     *
     * @param cName Clusterer name.
     * @param dset DataSet object.
     * @param k Neighborhood size.
     * @param distMat Distance matrix.
     * @param nsf NeighborSetFinder object for kNN sets.
     * @param trainingKernelMat Kernel matrix.
     * @param nsfKernel Kernel kNN object.
     * @param numClusters Number of clusters.
     * @param cmet CombinedMetric object for distance calculations.
     * @param ker Kernel object for kernel mappings.
     * @return
     */
    public static ClusteringAlg getClustererForName(
            String cName,
            DataSet dset,
            int k,
            float[][] distMat,
            NeighborSetFinder nsf,
            float[][] trainingKernelMat,
            NeighborSetFinder nsfKernel,
            int nClust,
            CombinedMetric cmet,
            Kernel ker) {
        ClusteringAlg clusterer = null;
        String algName = cName.toLowerCase();
        switch (algName) {
            case "ghpc":
            case "learning.unsupervised.methods.ghpc":
                clusterer = new GHPC(dset, cmet, nClust, k);
                ((GHPC) clusterer).setDistMatrix(distMat);
                ((GHPC) clusterer).setNSF(nsf);
                break;
            case "sup-ghpc":
                clusterer = new GHPC(dset, cmet, nClust, k);
                ((GHPC) clusterer).setDistMatrix(distMat);
                ((GHPC) clusterer).setHubnessMode(GHPC.SUPERVISED);
                ((GHPC) clusterer).setNSF(nsf);
                break;
            case "gkh":
            case "learning.unsupervised.methods.gkh":
                clusterer = new GKH(dset, cmet, nClust, k);
                ((GKH) clusterer).setDistMatrix(distMat);
                ((GKH) clusterer).setNSF(nsf);
                break;
            case "ghpkm":
            case "learning.unsupervised.methods.ghpkm":
                clusterer = new GHPKM(dset, cmet, nClust, k);
                ((GHPKM) clusterer).setDistMatrix(distMat);
                ((GHPKM) clusterer).setNSF(nsf);
                break;
            case "sup-ghpkm":
                clusterer = new GHPKM(dset, cmet, nClust, k);
                ((GHPKM) clusterer).setDistMatrix(distMat);
                ((GHPKM) clusterer).setHubnessMode(GHPC.SUPERVISED);
                ((GHPKM) clusterer).setNSF(nsf);
                break;
            case "kernel-ghpkm":
            case "ker-ghpkm":
            case "kernelghpkm":
            case "learning.unsupervised.methods.kernelghpkm":
                clusterer = new KernelGHPKM(dset, cmet, ker, nClust, k);
                ((KernelGHPKM) clusterer).setDistMatrix(distMat);
                ((KernelGHPKM) clusterer).setNSF(nsfKernel);
                break;
            case "lhpc":
            case "learning.unsupervised.methods.lhpc":
                clusterer = new LHPC(dset, cmet, nClust, k);
                ((LHPC) clusterer).setDistMatrix(distMat);
                break;
            case "lkh":
            case "learning.unsupervised.methods.lkh":
                clusterer = new LKH(dset, cmet, nClust, k);
                ((LKH) clusterer).setDistMatrix(distMat);
                break;
            case "kmeans":
            case "learning.unsupervised.methods.kmeans":
                clusterer = new KMeans(dset, cmet, nClust);
                break;
            case "kernel-kmeans":
            case "ker-kmeans":
            case "kernelkmeans":
            case "learning.unsupervised.methods.kernelkmeans":
                clusterer = new KernelKMeans(dset, ker, nClust);
                break;
            case "kmeansplusplus":
            case "learning.unsupervised.methods.kmeansplusplus":
                clusterer = new KMeansPlusPlus(dset, cmet, nClust);
                break;
            case "fastkmeans":
            case "learning.unsupervised.methods.fastkmeans":
                clusterer = new FastKMeans(dset, cmet, nClust);
                break;
            case "fastkmeansplusplus":
            case "learning.unsupervised.methods.fastkmeansplusplus":
                clusterer = new FastKMeansPlusPlus(dset, cmet, nClust);
                break;
            case "kmedoids":
            case "learning.unsupervised.methods.kmedoids":
                clusterer = new KMedoids(dset, cmet, nClust);
                break;
            case "kmedoidsplusplus":
            case "learning.unsupervised.methods.kmedoidsplusplus":
                clusterer = new KMedoidsPlusPlus(dset, cmet, nClust);
                break;
            case "harmonickmeans":
            case "learning.unsupervised.methods.harmonickmeans":
                clusterer = new HarmonicKMeans(dset, cmet, nClust);
                break;
            case "dbscan":
            case "learning.unsupervised.methods.dbscan" :
                clusterer = new DBScan(dset, cmet, 10);
                ((DBScan) clusterer).setNSF(nsf);
                break;
        }
        if (clusterer instanceof NSFUserInterface) {
            ((NSFUserInterface) clusterer).setNSF(nsf);
        }
        if (clusterer instanceof KernelMatrixUserInterface) {
            ((KernelMatrixUserInterface) clusterer).setKernelMatrix(
                    trainingKernelMat);
        }
        return clusterer;
    }
    
    /**
     * This method generates all supported clusterers representing different
     * algorithms, for the given data. It can be used in unit testing.
     *
     * @param dset DataSet object.
     * @param k Neighborhood size.
     * @param distMat Distance matrix.
     * @param nsf NeighborSetFinder object for kNN sets.
     * @param trainingKernelMat Kernel matrix.
     * @param nsfKernel Kernel kNN object.
     * @param numClusters Number of clusters.
     * @param cmet CombinedMetric object for distance calculations.
     * @param ker Kernel object for kernel mappings.
     * @return
     */
    public static ArrayList<ClusteringAlg> getAllClusterers(
            DataSet dset,
            int k,
            float[][] distMat,
            NeighborSetFinder nsf,
            float[][] trainingKernelMat,
            NeighborSetFinder nsfKernel,
            int nClust,
            CombinedMetric cmet,
            Kernel ker) {
        ArrayList<ClusteringAlg> allClusterers = new ArrayList<>(50);
        allClusterers.add(getClustererForName(
                "learning.unsupervised.methods.ghpc", dset, k, distMat, nsf,
                trainingKernelMat, nsfKernel, nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "sup-ghpc", dset, k, distMat, nsf, trainingKernelMat, nsfKernel,
                nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "learning.unsupervised.methods.gkh", dset, k, distMat, nsf,
                trainingKernelMat, nsfKernel, nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "learning.unsupervised.methods.ghpkm", dset, k, distMat, nsf,
                trainingKernelMat, nsfKernel, nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "sup-ghpkm", dset, k, distMat, nsf, trainingKernelMat,
                nsfKernel, nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "learning.unsupervised.methods.kernelghpkm", dset, k, distMat,
                nsf, trainingKernelMat, nsfKernel, nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "learning.unsupervised.methods.lhpc", dset, k, distMat, nsf,
                trainingKernelMat, nsfKernel, nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "learning.unsupervised.methods.lkh", dset, k, distMat, nsf,
                trainingKernelMat, nsfKernel, nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "learning.unsupervised.methods.kmeans", dset, k, distMat, nsf,
                trainingKernelMat, nsfKernel, nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "learning.unsupervised.methods.kernelkmeans", dset, k, distMat,
                nsf, trainingKernelMat, nsfKernel, nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "learning.unsupervised.methods.kmeansplusplus", dset, k,
                distMat, nsf, trainingKernelMat, nsfKernel, nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "learning.unsupervised.methods.fastkmeans", dset, k, distMat,
                nsf, trainingKernelMat, nsfKernel, nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "learning.unsupervised.methods.fastkmeansplusplus", dset, k,
                distMat, nsf, trainingKernelMat, nsfKernel, nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "learning.unsupervised.methods.kmedoids", dset, k, distMat, nsf,
                trainingKernelMat, nsfKernel, nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "learning.unsupervised.methods.kmedoidsplusplus", dset, k,
                distMat, nsf, trainingKernelMat, nsfKernel, nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "learning.unsupervised.methods.harmonickmeans", dset, k,
                distMat, nsf, trainingKernelMat, nsfKernel, nClust, cmet, ker));
        allClusterers.add(getClustererForName(
                "learning.unsupervised.methods.dbscan", dset, k, distMat, nsf,
                trainingKernelMat, nsfKernel, nClust, cmet, ker));
        return allClusterers;
    }
}
