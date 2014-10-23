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
package networked_experiments;

import data.representation.DataInstance;
import data.representation.DataSet;
import ioformat.IOARFF;
import java.io.BufferedReader;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashMap;
import org.openml.apiconnector.algorithms.TaskInformation;
import org.openml.apiconnector.io.ApiSessionHash;
import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.apiconnector.xml.DataSetDescription;
import org.openml.apiconnector.xml.Task;
import org.openml.apiconnector.xml.Task.Input.Data_set;
import org.openml.apiconnector.xml.Task.Input.Estimation_procedure;
import util.HTTPUtil;

/**
 * This class implements the basics functions that invoked the OpenML API in
 * order to run networked experiments, obtain the data and the folds from
 * OpenML, as well as upload the results to their service and make them
 * available for other researchers and comparisons.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HMOpenMLConnector {
    
    OpenmlConnector client;
    ApiSessionHash hashFetcher;
    
    /**
     * Initialization.
     * 
     * @param username String that is the OpenML authentication username.
     * @param password String that is the OpenML authentication password.
     */
    public HMOpenMLConnector(String username, String password) {
        client = new OpenmlConnector(username, password);
        hashFetcher = new ApiSessionHash(client);
        hashFetcher.set(username, password);
    }
    
    /**
     * This method fetches the data from OpenML without taking into account
     * potential ignore conditions for certain attributes. Therefore, it reads
     * in all of the data - which is later filtered.
     * 
     * @param taskId Integer that is the OpenML task ID.
     * @param targetFeatureName String that is the class feature name.
     * @return DataSet that is the loaded data.
     * @throws Exception 
     */
    private DataSet fetchUnfilteredDataSet(int taskId, String targetFeatureName)
            throws Exception {
        Task openmlTask = client.openmlTaskGet(taskId);
        Data_set dsObj = TaskInformation.getSourceData(openmlTask);
        DataSetDescription dataDescription =
                client.openmlDataDescription(dsObj.getData_set_id());
        String dataUrl = dataDescription.getUrl();
        String dsName = dataDescription.getName();
        String wholeDataString = HTTPUtil.get(dataUrl);
        IOARFF loader = new IOARFF();
        Reader in = new StringReader(wholeDataString);
        DataSet unfilteredDSet = null;
        try (BufferedReader br = new BufferedReader(in)) {
            unfilteredDSet = loader.load(br, targetFeatureName);
        } catch (Exception e) {
            throw e;
        }
        unfilteredDSet.setName(dsName);
        return unfilteredDSet;
    }
    
    /**
     * This method fetches all the train/test indexes from OpenML for
     * configuring the testing environment for classifier performance
     * estimation.
     * 
     * @param proc Estimation_procedure object to fetch the splits from.
     * @param numFolds Integer that is the number of folds.
     * @param numTimes Integer that is the number of times the fold split
     * process is repeated.
     * @return ArrayList<Integer>[][][] representing the train/test indexes.
     * @throws Exception 
     */
    private ArrayList<Integer>[][][] fetchAllTrainTestIndexes(
            Estimation_procedure proc,
            int numFolds,
            int numTimes)
            throws Exception {
        String trainTestUrl = proc.getData_splits_url();
        String trainTestString = HTTPUtil.get(trainTestUrl);
        IOARFF loader = new IOARFF();
        Reader in = new StringReader(trainTestString);
        DataSet trainTestIndexesDSet = null;
        try (BufferedReader br = new BufferedReader(in)) {
            trainTestIndexesDSet = loader.load(br);
        } catch (Exception e) {
            throw e;
        }
        int[] featSpecRepetition =
                trainTestIndexesDSet.getTypeAndIndexForAttrName("repeat");
        int[] featSpecFold =
                trainTestIndexesDSet.getTypeAndIndexForAttrName("fold");
        int[] featSpecInstanceIndex =
                trainTestIndexesDSet.getTypeAndIndexForAttrName("rowid");
        // All of the above are of the 'numeric' type, meaning that they are
        // read as floats even though they are integers. This is why we will
        // do rounding instead of the typical cast, just to be on the safe side.
        int fIndexRep = featSpecRepetition[1];
        int fIndexFold = featSpecFold[1];
        int fIndexInstance = featSpecInstanceIndex[1];
        int[] featSpecTrainTest =
                trainTestIndexesDSet.getTypeAndIndexForAttrName("type");
        int sIndexTT = featSpecTrainTest[1];
        ArrayList<Integer>[][][] trainTestIndexes = new ArrayList[numTimes][
                numFolds][2];
        for (int t = 0; t < numTimes; t++) {
            for (int f = 0; f < numFolds; f++) {
                trainTestIndexes[t][f][0] = new ArrayList<>();
                trainTestIndexes[t][f][1] = new ArrayList<>();
            }
        }
        int instanceIndex, repeatIndex, foldIndex;
        String ttSpec;
        for (int i = 0; i < trainTestIndexesDSet.size(); i++) {
            DataInstance indexingInstance = trainTestIndexesDSet.getInstance(i);
            instanceIndex = Math.round(indexingInstance.fAttr[fIndexInstance]);
            repeatIndex = Math.round(indexingInstance.fAttr[fIndexRep]);
            foldIndex = Math.round(indexingInstance.fAttr[fIndexFold]);
            ttSpec = indexingInstance.sAttr[sIndexTT];
            if (ttSpec.equalsIgnoreCase("TRAIN")) {
                trainTestIndexes[repeatIndex][foldIndex][0].add(instanceIndex);
            } else if (ttSpec.equalsIgnoreCase("TEST")) {
                trainTestIndexes[repeatIndex][foldIndex][1].add(instanceIndex);
            }
        }
        return trainTestIndexes;
    }
    
    /**
     * This method fetches all relevant data from OpenML for the given task.
     * 
     * @param taskId Integer that is the task ID.
     * @return DataFromOpenML object with all relevant data.
     * @throws Exception 
     */
    public DataFromOpenML fetchExperimentData(int taskId) throws Exception {
        hashFetcher.getSessionHash();
        System.out.println("OpenML data fetch initiated for taskID " + taskId);
        Task task = client.openmlTaskGet(taskId);
        Estimation_procedure est = TaskInformation.getEstimationProcedure(task);
        if (!est.getType().equalsIgnoreCase("crossvalidation")) {
            throw new Exception("The specified task estimation type is not "
                    + "cross-validation");
        }
        DataFromOpenML fetchedData = new DataFromOpenML();
        fetchedData.numFolds = TaskInformation.getNumberOfFolds(task);
        fetchedData.numTimes = TaskInformation.getNumberOfRepeats(task);
        fetchedData.classNames = TaskInformation.getClassNames(
                client, hashFetcher, task);
        // Not to confuse with Hub Miner's DataSet class.
        Data_set dsObj = TaskInformation.getSourceData(task);
        DataSetDescription dataDescription =
                dsObj.getDataSetDescription(client);
        fetchedData.targetFeatureName = dsObj.getTarget_feature();
        DataSet dset = fetchUnfilteredDataSet(taskId,
                fetchedData.targetFeatureName);
        ArrayList<String> allFeatures = dset.getAllFeatureNames();
        HashMap<String, String> ignoreMap = new HashMap<>(allFeatures.size());
        ArrayList<String> admissibleFeatures = new ArrayList<>();
        if (dataDescription.getIgnore_attribute() != null) {
            for(String attNameToIgnore : dataDescription.
                    getIgnore_attribute()) {
                ignoreMap.put(attNameToIgnore, attNameToIgnore);
            }
        }
        for (String attName: allFeatures) {
            if (attName.startsWith("'")) {
                attName = attName.substring(1);
            }
            if (attName.endsWith("'")) {
                attName = attName.substring(0,
                        attName.length() - 1);
            }
            if (!ignoreMap.containsKey(attName) && !attName.equals(
                    fetchedData.targetFeatureName)) {
                admissibleFeatures.add(attName);
            }
        }
        DataSet filteredDSet = DataSet.filterData(dset, admissibleFeatures);
        fetchedData.filteredDSet = filteredDSet;
        fetchedData.filteredDSet.setName(dset.getName());
        Estimation_procedure proc =
                TaskInformation.getEstimationProcedure(task);
        System.out.println("Starting to fetch train / test splits.");
        fetchedData.trainTestIndexes = fetchAllTrainTestIndexes(proc,
                fetchedData.numFolds, fetchedData.numTimes);
        System.out.println("Finished fetching train / test splits.");
        System.out.println("OpenML data fetch completed for taskID " + taskId);
        System.out.println("Dataset name: " + filteredDSet.getName());
        System.out.println("Data size: " + filteredDSet.size());
        System.out.println("Num float / int / nominal features: " +
                filteredDSet.getNumFloatAttr() + "," +
                filteredDSet.getNumIntAttr() + "," +
                filteredDSet.getNumNominalAttr());
        return fetchedData;
    }
    
    /**
     * @param taskId Integer that is the OpenML task ID.
     * @return String that is the dataset name corresponding to the provided
     * task id. 
     */
    public String getDataSetNameForTaskID(int taskId) throws Exception {
        Task openmlTask;
        DataSetDescription dataDescription = null;
        try {
            openmlTask = client.openmlTaskGet(taskId);
            Data_set dsObj = TaskInformation.getSourceData(openmlTask);
            dataDescription =
                    client.openmlDataDescription(dsObj.getData_set_id());
        } catch (Exception e) {
            System.err.println("Exception for taskID: " + taskId);
            throw e;
        }
        return dataDescription.getName();
    }
    
}
