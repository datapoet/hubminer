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
package visualization;

import com.google.gson.Gson;
import data.representation.DataSet;
import ioformat.SupervisedLoader;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.json.JSONObject;
import util.CommandLineParser;

/**
 * This class does the API calls to ViperCharts that generate useful
 * visualization charts of classifier performance.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ViperChartAPICall {
    
    private final String VIPERCHARTS_API_TARGET = "http://viper.ijs.si/api/";
    private int[] binaryDataLabels;
    private ArrayList<float[]> predictedDataLabels;
    private ArrayList<String> algorithmNames;
    private static final HashMap<ChartType, String> chartTypeToStringMap;
    private static final HashMap<String, ChartType> chartStringToTypeMap;
    private float[] precisionValues;
    private float[] recallValues;
    private float[] fprValues;
    private float[] tprValues;
    private boolean openInBrowser = false;
    private static final int DEFAULT_FONT_SIZE = 40;
    private int fontSize = DEFAULT_FONT_SIZE;
    private boolean embedded = false;
    
    static {
        chartTypeToStringMap = new HashMap<>();
        chartStringToTypeMap = new HashMap<>();
        chartTypeToStringMap.put(ChartType.PR_SPACE, "prs");
        chartStringToTypeMap.put("prs", ChartType.PR_SPACE);
        chartTypeToStringMap.put(ChartType.ROC_SPACE, "rocs");
        chartStringToTypeMap.put("rocs", ChartType.ROC_SPACE);
        chartTypeToStringMap.put(ChartType.PR_CURVES, "prc");
        chartStringToTypeMap.put("prc", ChartType.PR_CURVES);
        chartTypeToStringMap.put(ChartType.LIFT_CURVES, "lift");
        chartStringToTypeMap.put("lift", ChartType.LIFT_CURVES);
        chartTypeToStringMap.put(ChartType.ROC_CURVES, "rocc");
        chartStringToTypeMap.put("rocc", ChartType.ROC_CURVES);
        chartTypeToStringMap.put(ChartType.ROC_HULL_CURVES, "roch");
        chartStringToTypeMap.put("roch", ChartType.ROC_HULL_CURVES);
        chartTypeToStringMap.put(ChartType.COST_CURVES, "cost");
        chartStringToTypeMap.put("cost", ChartType.COST_CURVES);
        chartTypeToStringMap.put(ChartType.RATE_DRIVEN_CURVES, "ratedriven");
        chartStringToTypeMap.put("ratedriven", ChartType.RATE_DRIVEN_CURVES);
        chartTypeToStringMap.put(ChartType.KENDALL_CURVES, "kendall");
        chartStringToTypeMap.put("kendall", ChartType.KENDALL_CURVES);
    }
    
    /**
     * Initialization.
     * 
     * @param algorithmNames ArrayList<String> representing the algorithm names.
     * @param predictedDataLabels ArrayList<float[]> representing the predicted
     * class labels for each algorithm on the data.
     * @param binaryDataLabels int[] representing the ground truth labels.
     */
    public ViperChartAPICall(ArrayList<String> algorithmNames,
            ArrayList<float[]> predictedDataLabels, int[] binaryDataLabels,
            float[] recallValues, float[] precisionValues, float[] fprValues,
            float[] tprValues) {
        this.algorithmNames = algorithmNames;
        this.predictedDataLabels = predictedDataLabels;
        this.binaryDataLabels = binaryDataLabels;
        this.recallValues = recallValues;
        this.precisionValues = precisionValues;
        this.fprValues = fprValues;
        this.tprValues = tprValues;
    }
    
    /**
     * @param openInBrowser Boolean flag indicating whether to open the received
     * link in a browser.
     */
    public void openInBrowser(boolean openInBrowser) {
        this.openInBrowser = openInBrowser;
    }
    
    /**
     * This method makes the API call for the specified chart type.
     * 
     * @param cType ChartType to be calculated by ViperCharts.
     * @return String that is the URI of the generated chart, or an error
     * message.
     * @throws Exception 
     */
    public String generateChartForType(ChartType cType) throws Exception {
        int numAlgs = algorithmNames.size();
        JSONObject jObj = new JSONObject();
        jObj.put("chart", chartTypeToStringMap.get(cType));
        jObj.put("legend", "true");
        jObj.put("fontsize", fontSize);
        jObj.put("embedded", embedded);
        List jList = new ArrayList();
        if (cType == ChartType.PR_SPACE) {
            for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
                HashMap jMap = new HashMap();
                jList.add(jMap);
                jMap.put("name", algorithmNames.get(algIndex));
                jMap.put("recall", recallValues[algIndex]);
                jMap.put("precision", precisionValues[algIndex]);
            }
        } else if (cType == ChartType.ROC_SPACE) {
            for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
                HashMap jMap = new HashMap();
                jList.add(jMap);
                jMap.put("name", algorithmNames.get(algIndex));
                jMap.put("fpr", fprValues[algIndex]);
                jMap.put("tpr", tprValues[algIndex]);
            }
        } else {
            for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
                HashMap jMap = new HashMap();
                jList.add(jMap);
                jMap.put("name", algorithmNames.get(algIndex));
                jMap.put("predicted", predictedDataLabels.get(algIndex));
                jMap.put("actual", binaryDataLabels);
            }
        }
        jObj.put("data", jList);
        final String jsonStringRep = jObj.toString();
        System.out.println("Sending the following JSON request:" +
                jsonStringRep);
        byte[] jsonBytes = jsonStringRep.getBytes();
        URL u = new URL(VIPERCHARTS_API_TARGET);
        HttpURLConnection conn = (HttpURLConnection) u.openConnection();
        conn.setDoOutput(true);
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type",
                "application/json; charset=utf8");
        conn.setRequestProperty("Content-Length",
                String.valueOf(jsonBytes.length));
        conn.connect();
        try (OutputStream os = conn.getOutputStream()) {
            os.write(jsonStringRep.getBytes());
            os.flush();
        }
        InputStream response = conn.getInputStream();
        InputStreamReader reader = new InputStreamReader(response);
        String resultString;
        try (BufferedReader br = new BufferedReader(reader)) {
            resultString = br.readLine();
        }
        System.out.println("Server response: " + resultString);
        if (openInBrowser) {
            Gson gson = new Gson();
            HashMap responseMap = gson.fromJson(resultString, HashMap.class);
            if (responseMap.containsKey("url")) {
                String resultUrl = (String) responseMap.get("url");
                java.awt.Desktop.getDesktop().browse(java.net.URI.create(
                    resultUrl));
            }
        }
        return resultString;
    }
    
    /**
     * This method loads the average classifier predictions for data points from
     * a JSON file in the experimental result logs of Hub Miner classification
     * evaluation.
     * 
     * @param inFile File that is the input file containing the JSON log.
     * @param positiveClassIndex Integer that is the index of the class to use
     * as the positive class, since ViperCharts does charts for binary
     * classification tasks and we might be dealing with a multi-class problem.
     * @return float[] that is the predictions for the positive class.
     */
    private static float[] loadAverageProbabilisticClassifications(File inFile,
            int positiveClassIndex) throws IOException {
        if (!inFile.exists() || !inFile.isFile()) {
            throw new IOException("Invalid predictions load path.");
        }
        String jsonString = new String(Files.readAllBytes(
                Paths.get(inFile.getPath())));
        Gson gson = new Gson();
        float[][] predictionsAllClasses =
                gson.fromJson(jsonString, float[][].class);
        float[] predictionsForPositive = new float[
                predictionsAllClasses.length];
        for (int i = 0; i < predictionsAllClasses.length; i++) {
            predictionsForPositive[i] = predictionsAllClasses[i][
                    positiveClassIndex];
        }
        return predictionsForPositive;
    }
    
    /**
     * A script that generates a chart from the classification results.
     * 
     * @param args String[] that are the command line parameters, as specified.
     * @throws Exception 
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inAlgorithmDir", "Path to the algorithm data directory."
                + " This directory is named with the name of the algorithm, in"
                + "Hub Miner classification results. Also, this directory "
                + "contains the file avgProbClassAssignments.json that the "
                + "average point class prediction probabilites are to be read"
                + "from. Multiple algorithm prediction averages can be loaded"
                + "for chart construction.",
                CommandLineParser.STRING, true, true);
        clp.addParam("-inDataFile", "Path to the data file, to load the correct"
                + "classifications from, the ground truth.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-chartTypeStringCode", "String code indicating the chart"
                + "type, according to ViperCharts documentation.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-positiveClassIndex", "Index of the positive class.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-fontSize", "Display font size, in pt.",
                CommandLineParser.INTEGER, false, false);
        clp.addParam("-openInBrowser", "Whether to open the link immediately.",
                CommandLineParser.BOOLEAN, false, false);
        clp.addParam("-embedded", "If set to false, additional menues will be"
                + "loaded for configuring the chart. If set to true, just the"
                + "final chart will be embedded in the target page.",
                CommandLineParser.BOOLEAN, false, false);
        clp.parseLine(args);
        DataSet dset = SupervisedLoader.loadData((String) clp.getParamValues(
                "-inDataFile").get(0), false);
        int[] correctLabels = dset.obtainLabelArray();
        int positiveClassIndex = (Integer) clp.getParamValues(
                    "-positiveClassIndex").get(0);
        int numClasses = dset.countCategories();
        int numAlgs = clp.getParamValues("-inAlgorithmDir").size();
        int fontSize = clp.hasParamValue("-fontSize") ? (Integer)
                clp.getParamValues("-fontSize").get(0) : DEFAULT_FONT_SIZE;
        boolean openInBrowser = false;
        if (clp.hasParamValue("-openInBrowser")) {
            openInBrowser = (Boolean)
                    clp.getParamValues("-openInBrowser").get(0);
        }
        boolean embedded = false;
        if (clp.hasParamValue("-embedded")) {
            embedded = (Boolean) clp.getParamValues("-embedded").get(0);
        }
        File[] algDirs = new File[numAlgs];
        ArrayList<String> algNames = new ArrayList<>(numAlgs);
        ArrayList<float[]> predictedDataLabels = new ArrayList<>(numAlgs);
        int[] classFreqs = dset.getClassFrequencies();
        float precisionValue;
        float recallValue;
        float fprValue;
        float tprValue;
        float[] precisionValues = new float[numAlgs];
        float[] recallValues = new float[numAlgs];
        float[] fprValues = new float[numAlgs];
        float[] tprValues = new float[numAlgs];
        for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
            algDirs[algIndex] = new File((String) clp.getParamValues(
                    "-inAlgorithmDir").get(algIndex));
            File inPredictionsFile = new File(algDirs[algIndex],
                    "avgProbClassAssignments.json");
            float[] predictions = loadAverageProbabilisticClassifications(
                    inPredictionsFile, positiveClassIndex);
            predictedDataLabels.add(predictions);
            algNames.add(algDirs[algIndex].getName());
            File inAvgsFile = new File(algDirs[algIndex], "avg.txt");
            try (BufferedReader br = new BufferedReader(new InputStreamReader (
                    new FileInputStream(inAvgsFile)))) {
                // The header line.
                br.readLine();
                // The second line contains precision and recall values.
                String line = br.readLine();
                String[] lineItems = line.split(",");
                precisionValue = Float.parseFloat(lineItems[1]);
                recallValue = Float.parseFloat(lineItems[2]);
                line = br.readLine();
                while (line != null && !line.startsWith("confusion")) {
                    line = br.readLine();
                }
                // On the next line the confusion matrix starts.
                if (line == null) {
                    // This should not happen, unless the file has been manually
                    // corrupted.
                    fprValue = 0;
                    tprValue = 0;
                } else {
                    float[][] confusionMatrix = new float[numClasses][
                            numClasses];
                    for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                        line = br.readLine();
                        lineItems = line.split("\\s+");
                        for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                            confusionMatrix[cIndex][cSecond] =
                                    Float.parseFloat(lineItems[cSecond]);
                        }
                    }
                    float tpNum = confusionMatrix[positiveClassIndex][
                            positiveClassIndex];
                    int positiveClassSize = classFreqs[positiveClassIndex];
                    int negativeClassSize = dset.size() - positiveClassSize;
                    float fpNum = 0;
                    for (int cSecond = 0; cSecond < positiveClassIndex;
                            cSecond++) {
                        fpNum += confusionMatrix[positiveClassIndex][cSecond];
                    }
                    for (int cSecond = positiveClassIndex + 1;
                            cSecond < numClasses; cSecond++) {
                        fpNum += confusionMatrix[positiveClassIndex][cSecond];
                    }
                    fprValue = fpNum / negativeClassSize;
                    tprValue = tpNum / positiveClassSize;
                }
            }
            precisionValues[algIndex] = precisionValue;
            recallValues[algIndex] = recallValue;
            fprValues[algIndex] = fprValue;
            tprValues[algIndex] = tprValue;
        }
        String chartTypeStringCode = (String) clp.getParamValues(
                    "-chartTypeStringCode").get(0);
        ViperChartAPICall viperProxy = new ViperChartAPICall(algNames,
                predictedDataLabels, correctLabels, recallValues,
                precisionValues, fprValues, tprValues);
        viperProxy.fontSize = fontSize;
        viperProxy.embedded = embedded;
        ChartType cType = ChartType.ROC_CURVES;
        if (chartStringToTypeMap.containsKey(chartTypeStringCode)) {
            cType = chartStringToTypeMap.get(chartTypeStringCode);
        }
        viperProxy.openInBrowser(openInBrowser);
        viperProxy.generateChartForType(cType);
    }
}
