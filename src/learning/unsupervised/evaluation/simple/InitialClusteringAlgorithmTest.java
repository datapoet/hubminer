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
package learning.unsupervised.evaluation.simple;

import data.representation.DataInstance;
import data.representation.DataSet;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;

/**
 * Quickly test an algorithm to see if its output makes sense on a very simple
 * example.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class InitialClusteringAlgorithmTest {

    public static final int NUM_INSTANCES = 10000;

    /**
     * Tests an algorithm on 2D data and outputs the centroids
     *
     * @param clust Clusterer to test.
     * @param outFile Output file to write the centroids to.
     * @throws Exception
     */
    public static void test(ClusteringAlg clust, int numClusters, File outFile)
            throws Exception {
        String[] mn = new String[2];
        mn[0] = "X_att";
        mn[1] = "Y_att";
        DataSet dset = new DataSet(null, mn, null, NUM_INSTANCES);
        DataInstance[] randomSet = new DataInstance[NUM_INSTANCES];
        Random rGen = new Random();
        for (int i = 0; i < NUM_INSTANCES; i++) {
            randomSet[i] = new DataInstance(dset);
            randomSet[i].fAttr[0] = 1000 * rGen.nextFloat();
            randomSet[i].fAttr[1] = 1000 * rGen.nextFloat();
            dset.addDataInstance(randomSet[i]);
        }
        clust.setDataSet(dset);
        clust.setNumClusters(numClusters);
        clust.cluster();
        Cluster[] clusters = clust.getClusters();
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        try {
            for (int i = 0; i < clusters.length; i++) {
                DataInstance centroid = clusters[i].getCentroid();
                pw.print("centroid + " + i + " :");
                pw.print(centroid.fAttr[0]);
                pw.print(",");
                pw.print(centroid.fAttr[1]);
                pw.println();
            }
        } catch (Exception e) {
            pw.close();
        }
    }
}