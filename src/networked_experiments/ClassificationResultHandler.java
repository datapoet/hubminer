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

import com.thoughtworks.xstream.XStream;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import ioformat.IOARFF;
import ioformat.parsing.DataFeature;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import learning.supervised.evaluation.ValidateableInterface;
import org.apache.commons.lang3.StringUtils;
import org.openml.apiconnector.algorithms.Conversion;
import org.openml.apiconnector.algorithms.SciMark;
import org.openml.apiconnector.algorithms.TaskInformation;
import org.openml.apiconnector.io.ApiException;
import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.apiconnector.xml.Task.Output.Predictions.Feature;
import org.openml.apiconnector.xml.Implementation;
import org.openml.apiconnector.xml.Run;
import org.openml.apiconnector.xml.Run.Parameter_setting;
import org.openml.apiconnector.xml.Task;
import org.openml.apiconnector.xml.UploadRun;
import org.openml.apiconnector.xstream.XstreamXmlMapping;
import util.ArrayUtil;

/**
 * This class implement the methods that enable classification result upload to
 * OpenML servers.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClassificationResultHandler {
    
    // For API calls to OpenML.
    private final OpenmlConnector client;
    private final SciMark benchmarker;
    // For implementation registration with OpenML.
    private final File sourceCodeDir;
    private DataSet originalDset;
    private boolean useBenchmarker = false;
    
    /**
     * Initialization.
     * 
     * @param client OpenmlConnector for invoking OpenML API calls.
     * @param sourceCodeDir File that is the source code directory for the
     * algorithms used in the experiments.
     * @param originalDset DataSet that is the experiment data. 
     */
    public ClassificationResultHandler(OpenmlConnector client,
            File sourceCodeDir, DataSet originalDset) {
        this.client = client;
        this.benchmarker = new SciMark();
        this.sourceCodeDir = sourceCodeDir;
        this.originalDset = originalDset;
    }
    
    /**
     * @param useBenchmarker Boolean flag indicating whether to use the
     * benchmarker or not. It slows down result upload significantly.
     */
    public void setUseBenchmarker(boolean useBenchmarker) {
        this.useBenchmarker = useBenchmarker;
    }
    
    /**
     * This method prepares and uploads the classification results to the
     * OpenML servers.
     * 
     * @param task Task that is the OpenML task.
     * @param classifier ValidateableInterface which is the classifier to upload
     * the results for.
     * @param classifierIndex Integer that is the index of the classifier to
     * upload the results for.
     * @param parameterStringValues HashMap<String, String> mapping the
     * parameters of the classification algorithm to their values in the current
     * experiment run.
     * @param times Integer that is the number of repetitions in CV.
     * @param folds Integer that is the number of folds in CV.
     * @param foldTrainTestIndexes ArrayList<Integer>[][][] representing the
     * train/test splits for all repetitions and folds, as produced by OpenML
     * for the experiment.
     * @param allLabelAssignments float[][][][] representing all probabilistic
     * label assignments, for each algorithm, repetition and data point.
     */
    public void uploadClassificationResults(Task task,
            ValidateableInterface classifier, int classifierIndex,
            HashMap<String, String> parameterStringValues,
            int times, int folds, ArrayList<Integer>[][][] foldTrainTestIndexes,
            float[][][][] allLabelAssignments) throws Exception {
        OpenmlExecutedTask executedTask =
                new OpenmlExecutedTask(
                task,
                classifier,
                classifierIndex,
                parameterStringValues,
                client,
                times,
                folds,
                foldTrainTestIndexes,
                allLabelAssignments);
        Conversion.log("INFO", "Upload Run", "Starting send run process... ");
        if (useBenchmarker) {
            // The benchmarker tests JVM performance on the local machine, in
            // order to compare total execution times. It is a time-consuming
            // thing, so this should only be done when time is not an issue.
            executedTask.getRun().addOutputEvaluation("os_information",
                    "openml.userdefined.os_information(1.0)", null, "[" +
                    StringUtils.join(benchmarker.getOsInfo(), ", " ) + "]" );
            executedTask.getRun().addOutputEvaluation("scimark_benchmark",
                    "openml.userdefined.scimark_benchmark(1.0)",
                    benchmarker.getResult(), "[" + StringUtils.join(
                    benchmarker.getStringArray(), ", " ) + "]");
        }
        XStream xstream = XstreamXmlMapping.getInstance();
        // Save the classification predictions to a temporary file for later
        // upload.
        IOARFF pers = new IOARFF();
        String tmpPredictionsFileName = "classificationPredictions";
        File tmpPredictionsFile = File.createTempFile(tmpPredictionsFileName,
                ".arff");
        try (PrintWriter writer = new PrintWriter(new FileWriter(
                tmpPredictionsFile))) {
            pers.saveUnlabeled(executedTask.preparedPredictions, writer);
        }
        Map<String, File> outputFiles = new HashMap<>();
        outputFiles.put("predictions", tmpPredictionsFile);
        // Meta-information file.
        File tmpDescriptionFile = Conversion.stringToTempFile(xstream.toXML(
                executedTask.getRun()), "hubminer_generated_run", "xml");
        try { 
            UploadRun ur = client.openmlRunUpload(tmpDescriptionFile,
                    outputFiles);
            Conversion.log("INFO", "Upload Run", "Run was uploaded with rid "
                    + ur.getRun_id() + ". Obtainable at " +
                    client.getApiUrl() + "?f=openml.run.get&run_id=" + 
                    ur.getRun_id());
        } catch(ApiException ae) {
            System.err.println(ae.getMessage()); 
            Conversion.log("ERROR", "Upload Run", "Failed to upload run: " +
                    ae.getMessage());
        }
    }
    
    /**
     * This class stores the information about the completed classification task
     * under cross-validation and prepares the data for upload to OpenML
     * servers.
     */
    private class OpenmlExecutedTask {
        
        private String[] classNames;
        private Run run;
        private int implementationId;
        private DataSet preparedPredictions;

        /**
         * Initialization.
         * 
         * @param task Task that is the OpenML task.
         * @param classifier ValidateableInterface that is the classifier that
         * was tested.
         * @param classifierIndex Integer that is the index of the classifier
         * that is being evaluated.
         * @param parameterStringValues HashMap<String, String> that is a map of
         * parameter names and their values, as were used in the experiment.
         * @param client OpenmlConnector used for OpenML API calls.
         * @param times Integer that is the number of repetitions in the CV
         * framework.
         * @param folds Integer that is the number of folds in the CV framework.
         * @param foldTrainTestIndexes ArrayList<Integer>[][][] representing the
         * train and test index lists that were used in the experiments
         * @param allLabelAssignments float[][][][] representing all label
         * assignments.
         * @throws Exception 
         */
        public OpenmlExecutedTask(Task task, ValidateableInterface classifier,
                int classifierIndex,
                HashMap<String, String> parameterStringValues,
                OpenmlConnector client, int times, int folds,
                ArrayList<Integer>[][][] foldTrainTestIndexes,
                float[][][][] allLabelAssignments)
                throws Exception {
            if (allLabelAssignments == null) {
                throw new Exception("Label assignments not provided.");
            }
            if (foldTrainTestIndexes == null) {
                throw new Exception("Fold information not provided.");
            }
            if (originalDset == null || originalDset.isEmpty()) {
                throw new Exception("Experiment dataset not provided.");
            }
            classNames = TaskInformation.getClassNames(client, task);
            HashMap<String, Integer> classNameToIndexMap = new HashMap<>();
            for (int cIndex = 0; cIndex < classNames.length; cIndex++) {
                classNameToIndexMap.put(classNames[cIndex], cIndex);
            }
            // Generate an initial feature list.
            ArrayList<DataFeature> attInfo = new ArrayList<>();
            // Counters to incrementally determine the feature indexes within
            // their feature groups.
            int fFeatureIndex = -1;
            int sFeatureIndex = -1;
            // Lists of numeric and nominal feature names.
            ArrayList<String> fFeatNames = new ArrayList<>();
            ArrayList<String> sFeatNames = new ArrayList<>();
            // Start adding feature. First add the repetition / fold / instance
            // index features, which are numeric.
            DataFeature df = new DataFeature();
            df.setFeatureName("repeat");
            fFeatureIndex++;
            fFeatNames.add(df.getFeatureName());
            df.setFeatureIndex(fFeatureIndex);
            df.setFeatureType(DataMineConstants.FLOAT);
            attInfo.add(df);
            df = new DataFeature();
            df.setFeatureName("fold");
            fFeatureIndex++;
            fFeatNames.add(df.getFeatureName());
            df.setFeatureIndex(fFeatureIndex);
            df.setFeatureType(DataMineConstants.FLOAT);
            attInfo.add(df);
            df = new DataFeature();
            df.setFeatureName("row_id");
            fFeatureIndex++;
            fFeatNames.add(df.getFeatureName());
            df.setFeatureIndex(fFeatureIndex);
            df.setFeatureType(DataMineConstants.FLOAT);
            attInfo.add(df);
            // Start with the prediction features.
            for (Feature feat :
                    TaskInformation.getPredictions(task).getFeatures()) {
                switch (feat.getName()) {
                    case "confidence.classname":
                        for (String s : TaskInformation.getClassNames(
                                client, task)) {
                            df = new DataFeature();
                            df.setFeatureName("confidence." + s);
                            fFeatureIndex++;
                            df.setFeatureIndex(fFeatureIndex);
                            df.setFeatureType(DataMineConstants.FLOAT);
                            fFeatNames.add(df.getFeatureName());
                            attInfo.add(df);
                        }
                        break;
                    case "prediction":
                        sFeatureIndex++;
                        df = new DataFeature();
                        df.setFeatureName(feat.getName());
                        df.setFeatureIndex(sFeatureIndex);
                        df.setFeatureType(DataMineConstants.NOMINAL);
                        sFeatNames.add(df.getFeatureName());
                        attInfo.add(df);
                        break;
                }
            }
            // The feature for the correct data label.
            sFeatureIndex++;
            df = new DataFeature();
            df.setFeatureName("correct");
            df.setFeatureIndex(sFeatureIndex);
            df.setFeatureType(DataMineConstants.NOMINAL);
            sFeatNames.add(df.getFeatureName());
            attInfo.add(df);
            // Now prepare the results in the DataSet for upload.
            String[] fFeatNameArr = new String[fFeatNames.size()];
            fFeatNameArr = fFeatNames.toArray(fFeatNameArr);
            String[] sFeatNameArr = new String[sFeatNames.size()];
            sFeatNameArr = sFeatNames.toArray(sFeatNameArr);
            preparedPredictions = new DataSet();
            preparedPredictions.data = new ArrayList<>();
            preparedPredictions.fAttrNames = fFeatNameArr;
            preparedPredictions.sAttrNames = sFeatNameArr;
            // Generate prediction instances.
            for (int t = 0; t < times; t++) {
                for (int f = 0; f < folds; f++) {
                    for (int index: foldTrainTestIndexes[t][f][1]) {
                        DataInstance predictionInstance =
                                new DataInstance(preparedPredictions);
                        predictionInstance.embedInDataset(preparedPredictions);
                        for (DataFeature feature: attInfo) {
                            int featureIndex = feature.getFeatureIndex();
                            int featureType = feature.getFeatureType();
                            String featureName = feature.getFeatureName();
                            switch (featureType) {
                                case DataMineConstants.FLOAT:
                                {
                                    if (featureName.equals("repeat")) {
                                        predictionInstance.fAttr[
                                                featureIndex] = t;
                                    } else if (featureName.equals("fold")) {
                                        predictionInstance.fAttr[
                                                featureIndex] = f;
                                    } else if (featureName.equals("row_id")) {
                                        predictionInstance.fAttr[
                                                featureIndex] = index;
                                    } else if (featureName.startsWith(
                                            "confidence.")) {
                                        String className =
                                                featureName.substring(
                                                "confidence.".length());
                                        int classIndex =
                                                classNameToIndexMap.get(
                                                className);
                                        predictionInstance.fAttr[
                                                featureIndex] =
                                                allLabelAssignments[
                                                classifierIndex][t][index][
                                                classIndex];
                                    }
                                    break;
                                }
                                case DataMineConstants.NOMINAL:
                                {
                                    if (featureName.equals("prediction")) {
                                        // This is the prediction feature.
                                        float[] clPredictions =
                                                allLabelAssignments[
                                                classifierIndex][t][index];
                                        int clDecision = ArrayUtil.indexOfMax(
                                                clPredictions);
                                        predictionInstance.sAttr[featureIndex] =
                                                classNames[clDecision];
                                    } else if (featureName.equals("correct")) {
                                        predictionInstance.sAttr[featureIndex] =
                                                classNames[
                                                originalDset.getLabelOf(index)];
                                    }
                                    break;
                                }
                            }
                        }
                        preparedPredictions.addDataInstance(predictionInstance);
                    }
                }
            }
            // Register the implementation or obtain a registered implementation
            // based on an existing implementation ID.
            Implementation impAux = ClassifierRegistrationOpenML.create(
                    classifier.getClass(), parameterStringValues);
            implementationId = ClassifierRegistrationOpenML.getImplementationId(
                    impAux, classifier.getClass(), sourceCodeDir, client);
            Implementation impConfirmed = client.openmlImplementationGet(
                    implementationId);
            String setupString = classifier.getClass().getName();
            Set<String> paramNames = parameterStringValues.keySet();
            ArrayList<String> keyList = new ArrayList<>(paramNames.size());
            for (String pName: paramNames) {
                keyList.add(pName);
            }
            String[] keyArr = new String[keyList.size()];
            for (int i = 0; i < keyList.size(); i++) {
                keyArr[i] = keyList.get(i);
            }
            if(!parameterStringValues.isEmpty())
                setupString += (" -- " + keyArr);
            // Generate the Run object.
            List<Parameter_setting> list =
                    ClassifierRegistrationOpenML.getParameterSettingFromStrVals(
                    parameterStringValues, impConfirmed);
            run = new Run(task.getTask_id(), "", impConfirmed.getId(),
                    setupString, list.toArray(
                    new Parameter_setting[list.size()]) );
        }
        
        /**
         * @return Run of the experiments to upload. 
         */
        public Run getRun() {
            return run;
        }
    }
}
