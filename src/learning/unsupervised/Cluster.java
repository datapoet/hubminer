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

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.sparse.BOWDataSet;
import data.representation.sparse.BOWInstance;
import data.representation.util.DataInstanceDimComparator;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import ioformat.IOARFF;
import learning.unsupervised.evaluation.EmptyClusterException;
import util.ArrayUtil;
import util.DataSortUtil;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

/**
 * Implements the functionality for representing a data cluster.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Cluster implements Serializable {
    
    private static final long serialVersionUID = 1L;

    // The data context that the indexes point to.
    private DataSet dataContext = null;
    public ArrayList<Integer> indexes = null;

    /**
     * The default constructor.
     * @constructor
     */
    public Cluster() {
    }

    /**
     * @constructor @param dset Dataset the indexes will point to.
     */
    public Cluster(DataSet dset) {
        this.dataContext = dset;
        indexes = new ArrayList<>(Math.max(dset.size(), 10));
    }

    /**
     * @constructor @param dset Dataset the indexes will point to.
     * @param initSize Initial size of the index array.
     */
    public Cluster(DataSet dset, int initSize) {
        this.dataContext = dset;
        indexes = new ArrayList<>(Math.max(initSize, 10));
    }

    /**
     * @return Indexes of instances within the cluster.
     */
    public ArrayList<Integer> getIndexes() {
        return indexes;
    }

    /**
     * @param indexes Indexes of instances within the cluster.
     */
    public void setIndexes(ArrayList<Integer> indexes) {
        this.indexes = indexes;
    }

    /**
     * @return DataSet that the instances belong to.
     */
    public DataSet getDefinitionDataset() {
        return dataContext;
    }

    /**
     * @param DataSet that the instances belong to.
     */
    public void setDefinitionDataset(DataSet dset) {
        this.dataContext = dset;
    }

    /**
     * @param dset DataSet describing the data from within the cluster.
     * @param data ArrayList of data that is assigned to the current cluster.
     */
    public Cluster(DataSet dset, ArrayList<Integer> indexes) {
        this.dataContext = dset;
        this.indexes = indexes;
    }

    /**
     *
     * @param index Index within the cluster.
     * @return Index within the embedding dataset.
     */
    public int getWithinDataSetIndexOf(int index) {
        if (index < 0 || index > indexes.size()) {
            return -1;
        }
        return indexes.get(index);
    }

    /**
     * @param index Index within the cluster.
     * @param dsIndex Index within dataset.
     */
    public void setWithinDataSetIndexOf(int index, int dsIndex) {
        indexes.set(index, dsIndex);
    }

    /**
     * @return All data instances contained within the cluster.
     */
    public ArrayList<DataInstance> getAllInstances() {
        ArrayList<DataInstance> data =
                new ArrayList<>(indexes.size());
        for (int index : indexes) {
            data.add(dataContext.getInstance(index));
        }
        return data;
    }

    /**
     * @param index Index within the cluster.
     * @return Corresponding DataInstance.
     */
    public DataInstance getInstance(int index) {
        return dataContext.getInstance(indexes.get(index));
    }

    /**
     *
     * @return Cluster that contains the entire dataset.
     */
    public static Cluster fromEntireDataset(DataSet dset) {
        ArrayList<Integer> indexes = new ArrayList<>(dset.size());
        for (int i = 0; i < dset.size(); i++) {
            indexes.add(i);
        }
        return new Cluster(dset, indexes);
    }

    /**
     * @param index Index of the instance to be added to the cluster.
     */
    public void addInstance(int index) {
        indexes.add(index);
    }

    /**
     *
     * @param configuration Cluster configuration.
     * @param dataset Data set.
     * @return Cluster associations for a given clustering configuration.
     */
    public static int[] getAssociationsForClustering(Cluster[] configuration,
            DataSet dataset) {
        int[] clusterAssociations = new int[dataset.size()];
        for (int cIndex = 0; cIndex < configuration.length; cIndex++) {
            Cluster c = configuration[cIndex];
            for (int index : c.indexes) {
                clusterAssociations[index] = cIndex;
            }
        }
        return clusterAssociations;
    }

    /**
     * @param associations Integer array representing cluster associations.
     * @param dset Data set.
     * @return The cluster configuration array.
     */
    public static Cluster[] getConfigurationFromAssociations(int[] associations,
            DataSet dset) {
        int numClusters = ArrayUtil.max(associations) + 1;
        Cluster[] clusters = new Cluster[numClusters];
        for (int i = 0; i < numClusters; i++) {
            clusters[i] = new Cluster(dset);
        }
        if ((dset != null) && (associations != null)) {
            for (int i = 0; i < dset.size(); i++) {
                if (associations[i] >= 0) {
                    clusters[associations[i]].addInstance(i);
                }
            }
        }
        return clusters;
    }

    /**
     * @return DataInstance that is the cluster centroid.
     * @throws Exception
     */
    public DataInstance getCentroid() throws Exception {
        if (dataContext == null || dataContext.isEmpty() || isEmpty()) {
            throw new EmptyClusterException();
        }
        if (dataContext instanceof BOWDataSet) {
            return getCentroidSparse();
        } else {
            return getCentroidDense();
        }
    }

    /**
     * @return DataInstance that is the cluster centroid.
     * @throws Exception
     */
    private DataInstance getCentroidSparse() throws Exception {
        BOWInstance centroid = new BOWInstance((BOWDataSet) dataContext);
        // I use sparse vectors with headers usually, so the first element holds
        // the cardinality of the list.
        HashMap<Integer, Float> sparseSums = new HashMap<>(1000);
        HashMap<Integer, Integer> sparseCounts = new HashMap<>(1000);
        int[] integerCounts = new int[dataContext.getNumIntAttr()];
        int[] floatCounts = new int[dataContext.getNumFloatAttr()];
        float[] integerSums = new float[dataContext.getNumIntAttr()];
        float[] floatSums = new float[dataContext.getNumFloatAttr()];
        for (int i = 0; i < size(); i++) {
            BOWInstance instance = (BOWInstance) (getInstance(i));
            HashMap<Integer, Float> indexMap =
                    instance.getWordIndexesHash();
            Set<Integer> keys = indexMap.keySet();
            for (int index : keys) {
                float value = indexMap.get(index);
                if (DataMineConstants.isAcceptableFloat(value)) {
                    if (!sparseCounts.containsKey(index)) {
                        sparseCounts.put(index, 1);
                    } else {
                        sparseCounts.put(index, sparseCounts.get(index) + 1);
                    }
                    if (!sparseSums.containsKey(index)) {
                        sparseSums.put(index, 1f);
                    } else {
                        sparseSums.put(index, sparseSums.get(index) + 1);
                    }
                }
            }
            for (int j = 0; j < dataContext.getNumIntAttr(); j++) {
                if (DataMineConstants.isAcceptableInt(
                        getInstance(i).iAttr[j])) {
                    integerSums[j] += getInstance(i).iAttr[j];
                    integerCounts[j]++;
                }
            }
            for (int j = 0; j < dataContext.getNumFloatAttr(); j++) {
                if (DataMineConstants.isAcceptableFloat(
                        getInstance(i).fAttr[j])) {
                    floatSums[j] += getInstance(i).fAttr[j];
                    floatCounts[j]++;
                }
            }
        }
        if (dataContext.getNumNominalAttr() > 0) {
            for (int i = 0; i < dataContext.getNumNominalAttr(); i++) {
                centroid.sAttr[i] = "dummy" + i;
            }
        }
        for (int i = 0; i < dataContext.getNumIntAttr(); i++) {
            if (integerCounts[i] > 0) {
                centroid.iAttr[i] = (int) (integerSums[i] / integerCounts[i]);
            }
        }
        for (int i = 0; i < dataContext.getNumFloatAttr(); i++) {
            if (floatCounts[i] > 0) {
                centroid.fAttr[i] = floatSums[i] / floatCounts[i];
            }
        }
        Set<Integer> keys = sparseSums.keySet();
        for (int index : keys) {
            // These two map contain all the same keys
            if (sparseCounts.get(index) > 0) {
                sparseSums.put(index, sparseSums.get(index)
                        / sparseCounts.get(index));
            }
        }
        centroid.setWordIndexesHash(sparseSums);
        centroid.corpus = (BOWDataSet) dataContext;
        return centroid;
    }

    /**
     * @throws Exception
     * @return DataInstance that is the cluster centroid.
     */
    private DataInstance getCentroidDense() throws Exception {
        DataInstance centroid = new DataInstance(dataContext);
        int[] integerCounts = new int[dataContext.getNumIntAttr()];
        int[] floatCounts = new int[dataContext.getNumFloatAttr()];
        float[] integerSums = new float[dataContext.getNumIntAttr()];
        float[] floatSums = new float[dataContext.getNumFloatAttr()];
        for (int i = 0; i < size(); i++) {
            for (int j = 0; j < dataContext.getNumIntAttr(); j++) {
                if (DataMineConstants.isAcceptableInt(
                        getInstance(i).iAttr[j])) {
                    integerSums[j] += getInstance(i).iAttr[j];
                    integerCounts[j]++;
                }
            }
            for (int j = 0; j < dataContext.getNumFloatAttr(); j++) {
                if (DataMineConstants.isAcceptableFloat(
                        getInstance(i).fAttr[j])) {
                    floatSums[j] += getInstance(i).fAttr[j];
                    floatCounts[j]++;
                }
            }
        }
        if (dataContext.getNumNominalAttr() > 0) {
            for (int i = 0; i < dataContext.getNumNominalAttr(); i++) {
                centroid.sAttr[i] = "dummy" + i;
            }
        }
        for (int i = 0; i < dataContext.getNumIntAttr(); i++) {
            if (integerCounts[i] > 0) {
                centroid.iAttr[i] = (int) (integerSums[i] / integerCounts[i]);
            }
        }
        for (int i = 0; i < dataContext.getNumFloatAttr(); i++) {
            if (floatCounts[i] > 0) {
                centroid.fAttr[i] = floatSums[i] / floatCounts[i];
            }
        }
        return centroid;
    }

    /**
     * @throws Exception
     * @return DataInstance closest to the cluster centroid.
     */
    public DataInstance getMedoid() throws Exception {
        return getMedoid(CombinedMetric.EUCLIDEAN);
    }

    /**
     * @param cmet CombinedMetric for calculating the distances.
     * @throws Exception
     * @return DataInstance closest to the cluster centroid.
     */
    public DataInstance getMedoid(CombinedMetric cmet) throws Exception {
        if (dataContext == null || dataContext.isEmpty() || isEmpty()) {
            throw new EmptyClusterException();
        }
        DataInstance centroid = getCentroid();
        DataInstance medoid = getInstance(0);
        float currDistance = cmet.dist(centroid, medoid);
        if (!DataMineConstants.isAcceptableFloat(currDistance)) {
            currDistance = Float.MAX_VALUE;
        }
        float minDistance = currDistance;
        for (int i = 1; i < size(); i++) {
            currDistance = cmet.dist(centroid, getInstance(i));
            if (!DataMineConstants.isAcceptableFloat(currDistance)) {
                currDistance = Float.MAX_VALUE;
            }
            if (currDistance < minDistance) {
                minDistance = currDistance;
                medoid = getInstance(i);
            }
        }
        return medoid;
    }

    /**
     * @param cmet CombinedMetric for calculating the distances.
     * @throws Exception
     * @return Index of the DataInstance closest to the cluster centroid.
     */
    public int getMedoidIndex(CombinedMetric cmet) throws Exception {
        if (dataContext == null || dataContext.isEmpty() || isEmpty()) {
            throw new EmptyClusterException();
        }
        DataInstance centroid = getCentroid();
        int medoidIndex = indexes.get(0);
        float currDistance = cmet.dist(centroid,
                dataContext.getInstance(medoidIndex));
        if (!DataMineConstants.isAcceptableFloat(currDistance)) {
            currDistance = Float.MAX_VALUE;
        }
        float minDistance = currDistance;
        for (int i = 1; i < size(); i++) {
            currDistance = cmet.dist(centroid, getInstance(i));
            if (!DataMineConstants.isAcceptableFloat(currDistance)) {
                currDistance = Float.MAX_VALUE;
            }
            if (currDistance < minDistance) {
                minDistance = currDistance;
                medoidIndex = indexes.get(i);
            }
        }
        return medoidIndex;
    }

    /**
     * @param featureIndex Integer index of the feature.
     * @param featureType Feature type: floats or ints.
     * @throws Exception
     * @return The featureMedian value for the desired data feature.
     */
    public float getMedianForDimension(int featureIndex, int featureType)
            throws Exception {
        if (dataContext == null || dataContext.isEmpty() || isEmpty()) {
            return 0;
        }
        if (featureIndex < 0
                || (featureType == DataMineConstants.FLOAT
                && featureIndex > dataContext.getNumFloatAttr())
                || (featureType == DataMineConstants.INTEGER
                && featureIndex > dataContext.getNumIntAttr())) {
            throw new Exception("Non-negative index within bounds required.");
        }
        float featureMedian = 0;
        switch (featureType) {
            case DataMineConstants.FLOAT: {
                DataInstanceDimComparator dataComparator =
                        new DataInstanceDimComparator(
                        featureIndex, featureType);
                ArrayList<DataInstance> data = getAllInstances();
                DataSortUtil.sort(data, dataComparator);
                if (data.size() % 2 == 0) {
                    featureMedian =
                            ((data.get(data.size() / 2)).fAttr[featureIndex]
                            + (data.get((data.size() / 2) - 1)).fAttr[
                            featureIndex]) / 2;
                } else {
                    featureMedian =
                            (data.get(data.size() / 2)).fAttr[featureIndex];
                }
                break;
            }
            case DataMineConstants.INTEGER: {
                DataInstanceDimComparator dataComparator =
                        new DataInstanceDimComparator(
                        featureIndex, featureType);
                ArrayList<DataInstance> data = getAllInstances();
                DataSortUtil.sort(data, dataComparator);
                if (data.size() % 2 == 0) {
                    featureMedian =
                            ((data.get(data.size() / 2)).iAttr[featureIndex]
                            + (data.get((data.size() / 2) - 1)).iAttr[
                            featureIndex]) / 2;
                } else {
                    featureMedian =
                            (data.get(data.size() / 2)).iAttr[featureIndex];
                }
                break;
            }
        }
        return featureMedian;
    }

    /**
     * @throws Exception
     * @return Diameter of the cluster.
     */
    public float calculateDiameter() throws Exception {
        DataInstance centroid = getCentroid();
        return calculateDiameter(centroid);
    }

    /**
     * @param cmet CombinedMetric used to calculate the distances.
     * @throws Exception
     * @return Diameter of the cluster.
     */
    public float calculateDiameter(CombinedMetric cmet) throws Exception {
        DataInstance centroid = getCentroid();
        return calculateDiameter(centroid, cmet);
    }

    /**
     * @param DataInstance that is the Cluster centroid.
     * @throws Exception
     * @return Diameter of the cluster.
     */
    public float calculateDiameter(DataInstance centroid) throws Exception {
        return calculateDiameter(centroid, CombinedMetric.EUCLIDEAN);
    }

    /**
     * @param cmet CombinedMetric used to calculate the distances.
     * @param DataInstance that is the Cluster centroid.
     * @throws Exception
     * @return Diameter of the cluster.
     */
    public float calculateDiameter(DataInstance centroid, CombinedMetric cmet)
            throws Exception {
        if (dataContext == null || dataContext.isEmpty() || isEmpty()) {
            return 0;
        }
        if (cmet == null) {
            throw new Exception("Metrics object required.");
        }
        if (centroid == null) {
            throw new Exception("Centroid required.");
        }
        float clusterDiameter = 0;
        for (DataInstance instance : getAllInstances()) {
            clusterDiameter = Math.max(clusterDiameter,
                    cmet.dist(centroid, instance));
        }
        return clusterDiameter;
    }

    /**
     * @throws Exception
     * @return Average distance between data instances within the cluster.
     */
    public float averageIntraDistance() throws Exception {
        DataInstance centroid = getCentroid();
        return averageIntraDistance(centroid);
    }

    /**
     * @param DataInstance that is the Cluster centroid.
     * @throws Exception
     * @return Average distance between data instances within the cluster.
     */
    public float averageIntraDistance(DataInstance centroid) throws Exception {
        return averageIntraDistance(centroid, CombinedMetric.EUCLIDEAN);
    }

    /**
     * @param cmet CombinedMetric object.
     * @throws Exception
     * @return Average distance between data instances within the cluster.
     */
    public float averageIntraDistance(CombinedMetric cmet) throws Exception {
        DataInstance centroid = getCentroid();
        return averageIntraDistance(centroid, cmet);
    }

    /**
     * @param DataInstance that is the Cluster centroid.
     * @param cmet CombinedMetric object.
     * @throws Exception
     * @return Average distance between data instances within the cluster.
     */
    public float averageIntraDistance(DataInstance centroid,
            CombinedMetric cmet) throws Exception {
        if (dataContext == null || dataContext.isEmpty() || isEmpty()) {
            return 0;
        }
        if (cmet == null) {
            throw new Exception("Metrics object required.");
        }
        if (centroid == null) {
            throw new Exception("Centroid required.");
        }
        float distance = 0;
        for (DataInstance instance : getAllInstances()) {
            distance += cmet.dist(centroid, instance);
        }
        return distance / size();
    }

    /**
     * @return True in case of an empty cluster, false otherwise.
     */
    public boolean isEmpty() {
        return (size() == 0);
    }

    /**
     * @return A new cluster that is a copy of the current cluster.
     */
    public Cluster copy() {
        if (dataContext == null || dataContext.isEmpty() || isEmpty()) {
            return new Cluster(dataContext);
        }
        Cluster newCluster = new Cluster(dataContext);
        newCluster.indexes = new ArrayList<>(size());
        for (int i = 0; i < size(); i++) {
            newCluster.indexes.add(indexes.get(i));
        }
        return newCluster;
    }

    /**
     * Loads the cluster configuration from a file.
     *
     * @param f File where the configuration is to be loaded from.
     * @return
     */
    public static Cluster[] loadConfigurationFromFile(File f) {
        Cluster[] configuration;
        DataSet dset = null;
        IOARFF saverloader = new IOARFF();
        try {
            dset = saverloader.load(f.getPath());
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
        if (dset == null) {
            return null;
        }
        // See how many clusters there are and make an empty configuration.
        int max = -1;
        for (int i = 0; i < dset.size(); i++) {
            max = Math.max(max,
                    dset.getInstance(i).iAttr[
                        dset.getInstance(i).iAttr.length - 1]);
        }
        if (max == -1) {
            return null;
        }
        configuration = new Cluster[max + 1];
        // Make a new dataset without the cluster number.
        DataSet newDS = new DataSet();
        newDS.fAttrNames = dset.fAttrNames;
        newDS.sAttrNames = dset.sAttrNames;
        if (dset.getNumIntAttr() == 1) {
            newDS.iAttrNames = null;
        } else {
            newDS.iAttrNames = new String[dset.getNumIntAttr() - 1];
            for (int i = 0; i < dset.getNumIntAttr() - 1; i++) {
                newDS.iAttrNames[i] = dset.iAttrNames[i];
            }
        }
        // Initialize clusters.
        for (int i = 0; i < configuration.length; i++) {
            configuration[i] = new Cluster(newDS);
        }
        DataInstance newInstance;
        for (int i = 0; i < dset.size(); i++) {
            newInstance = new DataInstance();
            newInstance.embedInDataset(newDS);
            newDS.addDataInstance(newInstance);
            newInstance.fAttr = dset.getInstance(i).fAttr;
            newInstance.sAttr = dset.getInstance(i).sAttr;
            if (newDS.iAttrNames != null) {
                newInstance.iAttr = new int[newDS.iAttrNames.length];
                for (int j = 0; j < newDS.iAttrNames.length; j++) {
                    newInstance.iAttr[j] = dset.getInstance(i).iAttr[j];
                }
            }
            configuration[dset.getInstance(i).iAttr[
                    dset.getInstance(i).iAttr.length - 1]].addInstance(i);
        }
        return configuration;
    }

    /**
     * Write the cluster configuration to a file.
     *
     * @param f File where the configuration is to be written.
     * @param configuration Cluster configuration
     * @param dset Data set.
     */
    public static void writeConfigurationToFile(File f,
            Cluster[] configuration, DataSet dset) {
        if (f == null || f.isDirectory() || configuration == null
                || configuration.length == 0) {
            return;
        }
        try (PrintWriter pw = new PrintWriter(new FileWriter(f));) {
            // Set header.
            pw.println("@RELATION ClusterConfiguration");
            for (int i = 0; i < dset.getNumIntAttr(); i++) {
                pw.println("@ATTRIBUTE " + dset.iAttrNames[i] + " integer");
            }
            pw.println("@ATTRIBUTE " + "cluster" + " integer");
            for (int i = 0; i < dset.getNumFloatAttr(); i++) {
                pw.println("@ATTRIBUTE " + dset.fAttrNames[i] + " real");
            }
            for (int i = 0; i < dset.getNumNominalAttr(); i++) {
                pw.println("@ATTRIBUTE " + dset.sAttrNames[i] + " string");
            }
            pw.println("@DATA");
            for (int i = 0; i < configuration.length; i++) {
                if (configuration[i] != null && !configuration[i].isEmpty()) {
                    for (int j = 0; j < configuration[i].size(); j++) {
                        boolean first = true;
                        if (dset.iAttrNames != null) {
                            for (int k = 0; k < dset.getNumIntAttr(); k++) {
                                if (!first) {
                                    pw.print(","
                                            + configuration[i].getInstance(
                                            j).iAttr[k]);
                                } else {
                                    pw.print(configuration[i].getInstance(
                                            j).iAttr[k]);
                                    first = false;
                                }
                            }
                            if (!first) {
                                pw.print("," + i);
                            } else {
                                pw.print(i);
                                first = false;
                            }
                        } else {
                            if (!first) {
                                pw.print("," + i);
                            } else {
                                pw.print(i);
                                first = false;
                            }
                        }
                        for (int k = 0; k < dset.getNumFloatAttr(); k++) {
                            if (!first) {
                                pw.print(","
                                        + configuration[i].getInstance(
                                        j).fAttr[k]);
                            } else {
                                pw.print(configuration[i].getInstance(
                                        j).fAttr[k]);
                                first = false;
                            }
                        }
                        for (int k = 0; k < dset.getNumNominalAttr(); k++) {
                            if (!first) {
                                pw.print(","
                                        + configuration[i].getInstance(
                                        j).sAttr[k]);
                            } else {
                                pw.print(configuration[i].getInstance(
                                        j).sAttr[k]);
                                first = false;
                            }
                        }
                        pw.println();
                    }
                }
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     * @return Size of the cluster.
     */
    public int size() {
        if (indexes == null) {
            return 0;
        } else {
            return indexes.size();
        }
    }

    /**
     * Transforms the cluster into a DataSet of its own.
     *
     * @return
     */
    public DataSet intoDataSet() {
        DataSet container = dataContext.cloneDefinition();
        container.data = new ArrayList<>(size());
        for (int i = 0; i < size(); i++) {
            container.addDataInstance(getInstance(i));
        }
        return container;
    }

    /**
     * Extract data classes as cluster objects.
     *
     * @param dset DataSet object.
     * @return An array of clusters representing data classes.
     * @throws Exception
     */
    public static Cluster[] getClustersFromCategorizedData(
            DataSet dset) throws Exception {
        Cluster[] config;
        if (dset.isEmpty()) {
            return null;
        }
        // Find biggest category index.
        int maxCat = -1;
        for (DataInstance instance : dset.data) {
            if (instance.getCategory() > maxCat) {
                maxCat = instance.getCategory();
            }
        }
        int numCat = maxCat + 1;
        config = new Cluster[numCat];
        for (int i = 0; i < numCat; i++) {
            config[i] = new Cluster(dset);
        }
        for (int i = 0; i < dset.size(); i++) {
            config[dset.getLabelOf(i)].addInstance(i);
        }
        return config;
    }

    /**
     * @param index The index of the desired index within the indexes ArrayList
     * in the Cluster object.
     * @return Index of DataInstance in the embedding DataSet object that is
     * located at position equal to the passed index parameter in the Cluster
     * ArrayList of indexes.
     */
    public int getIndex(int index) {
        if (indexes == null || index < 0 || index > indexes.size()) {
            return -1;
        } else {
            return indexes.get(index);
        }
    }
}