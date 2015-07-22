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
package learning.unsupervised.outliers;

import java.util.ArrayList;
import java.util.HashMap;

import data.representation.DataSet;
import data.representation.DataInstance;

/**
 * Class that defines the methods for outlier detection and implements some
 * common functions.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public abstract class OutlierDetector {

    private ArrayList<Integer> outlierIndexes;
    private HashMap<Integer, Float> outlierScoresMap;
    private DataSet dataset;

    /**
     * @return The number of outliers that have been detected.
     */
    public int numberOfOutliersFound() {
        return outlierIndexes == null ? 0 : outlierIndexes.size();
    }
    
    /**
     * @return HashMap<Integer, Float> that maps the outlier indexes to their 
     * calculated outlier scores. If the point is not an outlier, its index is 
     * not in the map.
     */
    public HashMap<Integer, Float> getOutlierScoresMap() {
        return outlierScoresMap;
    }

    /**
     * Checks if an index points to an outlier instance.
     *
     * @param index Index within a DataSet.
     * @return True if the instance is an outlier, false otherwise.
     */
    public boolean isOutlier(int index) {
        return outlierScoresMap
                == null ? false : outlierScoresMap.containsKey(index);
    }

    /**
     * Sets the outlier indexes and outlier scores. Meant to be used from within
     * the classes that extend OutlierDetector.
     *
     * @param outlierIndexes The indexes of detected outliers.
     * @param outlierScores The calculated outlier scores.
     */
    protected void setOutlierIndexes(
            ArrayList<Integer> outlierIndexes, ArrayList<Float> outlierScores) {
        if (outlierIndexes == null) {
            return;
        }
        this.outlierIndexes = outlierIndexes;
        outlierScoresMap = new HashMap<>(outlierIndexes.size());
        for (int i = 0; i < outlierIndexes.size(); i++) {
            outlierScoresMap.put(outlierIndexes.get(i), outlierScores.get(i));
        }
    }

    /**
     * Sets the outlier indexes. Meant to be used from within the classes that
     * extend OutlierDetector.
     *
     * @param outlierIndexes
     */
    protected void setOutlierIndexes(ArrayList<Integer> outlierIndexes) {
        if (outlierIndexes == null) {
            return;
        }
        this.outlierIndexes = outlierIndexes;
        outlierScoresMap = new HashMap<>(outlierIndexes.size());
        for (int i = 0; i < outlierIndexes.size(); i++) {
            outlierScoresMap.put(outlierIndexes.get(i), 1f);
        }
    }

    /**
     * @return The indexes of the detected outliers within the DataSet.
     */
    public ArrayList<Integer> getOutlierIndexes() {
        return outlierIndexes;
    }

    /**
     * Sets the dataset to be analyzed.
     *
     * @param dataset DataSet object.
     */
    public void setDataSet(DataSet dataset) {
        this.dataset = dataset;
    }

    /**
     * @return DataSet that is currently being analyzed.
     */
    public DataSet getDataSet() {
        return this.dataset;
    }

    /**
     * Gets the actual outliers instead of just the indexes.
     *
     * @return A list of outlier objects.
     */
    public ArrayList<DataInstance> getOutliers() {
        if (outlierIndexes == null) {
            return new ArrayList<>();
        }
        ArrayList<DataInstance> outliers =
                new ArrayList<>(outlierIndexes.size());
        for (int index : outlierIndexes) {
            outliers.add(dataset.getInstance(index));
        }
        return outliers;
    }

    /**
     * @throws Exception
     */
    public abstract void detectOutliers() throws Exception;

    /**
     * Here we first search for outliers and then remove them.
     *
     * @return DataSet cleaned of outliers, a new object.
     * @throws Exception
     */
    public DataSet findAndRemoveOutliers() throws Exception {
        detectOutliers();
        return removeFoundOutliers();
    }

    /**
     * Here we remove already detected outliers
     *
     * @return DataSet cleaned of outliers, a new object.
     * @throws Exception
     */
    public DataSet removeFoundOutliers() throws Exception {
        if (numberOfOutliersFound() == 0) {
            return dataset;
        }
        DataSet cleanedData = dataset.cloneDefinition();
        for (int i = 0; i < dataset.size(); i++) {
            if (!isOutlier(i)) {
                DataInstance instance = dataset.getInstance(i).copy();
                instance.embedInDataset(cleanedData);
                cleanedData.addDataInstance(instance);
            }
        }
        return cleanedData;
    }
}