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
        chartTypeToStringMap.put(ChartType.COLUMN_CHART, "column");
        chartStringToTypeMap.put("column", ChartType.COLUMN_CHART);
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
            ArrayList<float[]> predictedDataLabels, int[] binaryDataLabels) {
        this.algorithmNames = algorithmNames;
        this.predictedDataLabels = predictedDataLabels;
        this.binaryDataLabels = binaryDataLabels;
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
        JSONObject jobj = new JSONObject();
        jobj.put("chart", chartTypeToStringMap.get(cType));
        List jList = new ArrayList();
        for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
            HashMap jMap = new HashMap();
            jList.add(jMap);
            jMap.put("name", algorithmNames.get(algIndex));
            jMap.put("predicted", predictedDataLabels.get(algIndex));
            jMap.put("actual", binaryDataLabels);
        }
        jobj.put("data", jList);
        final String jsonStringRep = jobj.toString();
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
        clp.parseLine(args);
        DataSet dset = SupervisedLoader.loadData((String) clp.getParamValues(
                "-inDataFile").get(0), false);
        int[] correctLabels = dset.obtainLabelArray();
        int positiveClassIndex = (Integer) clp.getParamValues(
                    "-positiveClassIndex").get(0);
        int numAlgs = clp.getParamValues("-inAlgorithmDir").size();
        File[] algDirs = new File[numAlgs];
        ArrayList<String> algNames = new ArrayList<>(numAlgs);
        ArrayList<float[]> predictedDataLabels = new ArrayList<>(numAlgs);
        for (int algIndex = 0; algIndex < numAlgs; algIndex++) {
            algDirs[algIndex] = new File((String) clp.getParamValues(
                    "-inAlgorithmDir").get(algIndex));
            File inFile = new File(algDirs[algIndex],
                    "avgProbClassAssignments.json");
            float[] predictions = loadAverageProbabilisticClassifications(
                    inFile, positiveClassIndex);
            predictedDataLabels.add(predictions);
            algNames.add(algDirs[algIndex].getName());
        }
        String chartTypeStringCode = (String) clp.getParamValues(
                    "-chartTypeStringCode").get(0);
        ViperChartAPICall viperProxy = new ViperChartAPICall(algNames,
                predictedDataLabels, correctLabels);
        ChartType cType = ChartType.ROC_CURVES;
        if (chartStringToTypeMap.containsKey(chartTypeStringCode)) {
            cType = chartStringToTypeMap.get(chartTypeStringCode);
        }
        viperProxy.generateChartForType(cType);
    }
}
