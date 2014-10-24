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

import java.io.File;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;
import learning.supervised.evaluation.ValidateableInterface;
import org.openml.apiconnector.algorithms.Conversion;
import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.apiconnector.xml.Implementation;
import org.openml.apiconnector.xml.Implementation.Parameter;
import org.openml.apiconnector.xml.ImplementationExists;
import org.openml.apiconnector.xml.Run.Parameter_setting;
import org.openml.apiconnector.xml.UploadImplementation;
import org.openml.apiconnector.xstream.XstreamXmlMapping;

/**
 * This class implements the classifier registration and parameter registration
 * for experiments used with OpenML-fetched data, in order to be able to upload
 * the results of the experiments back to the servers.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClassifierRegistrationOpenML {
    
    /**
     * This method produces the current algorithm version for the OpenML
     * archive. It currently relies on the serial UID, which is an initial
     * option. TODO: Introduce more precise version control.
     * 
     * @param classifier ValidateableInterface that is the classifier used in
     * OpenML-driven experiments.
     * @return String that is the version string of the classifier.
     */
    public String getVersionString(ValidateableInterface classifier) {
        return Long.toString(classifier.getVersion());
    }
    
    /**
     * This method returns the implementation ID from the OpenML servers, if it
     * was registered.
     * 
     * @param implementation Implementation corresponding the the classifier.
     * @param classifier classifierClass Class of the algorithm used in
     * classification.
     * @param client OpenmlConnector client for communicating with OpenML.
     * @return Integer that is the implementation ID obtained from the OpenML
     * servers.
     * @throws Exception 
     */
    public static int getImplementationId(Implementation implementation,
            Class classifierClass, File hubMinerSourceDir,
            OpenmlConnector client)
            throws Exception {
        if (classifierClass == null) {
            throw new Exception("Null algorithm class provided.");
        }
        ImplementationExists result;
        try {
            result = client.openmlImplementationExists(
                    implementation.getName(),
                    implementation.getExternal_version());
        } catch (Exception e) {
            return registerImplementation(implementation, classifierClass,
                    hubMinerSourceDir, client);
        }
        if(result != null && result.exists()) {
            return result.getId();
        } else {
            return registerImplementation(implementation, classifierClass,
                    hubMinerSourceDir, client);
        }
    }
    
    /**
     * This method registers an algorithm and returns the implementation ID from
     * the OpenML servers was registered. It assumes that the implementation
     * was not previously registered, so it is a privat method used internally.
     * 
     * @param implementation Implementation corresponding the the classifier.
     * @param classifier classifierClass Class of the algorithm used in
     * classification.
     * @param client OpenmlConnector client for communicating with OpenML.
     * @return Integer that is the implementation ID obtained from the OpenML
     * servers.
     * @throws Exception 
     */
    private static int registerImplementation(Implementation implementation,
            Class classifierClass, File hubMinerSourceDir,
            OpenmlConnector client)
            throws Exception {
        String xml = XstreamXmlMapping.getInstance().toXML(implementation);
        File implementationFile = Conversion.stringToTempFile(
                xml, implementation.getName(), "xml");
        String canonicalClassifierName = classifierClass.getCanonicalName();
        String pathTail = canonicalClassifierName.replace(".",
                File.separator).concat(".java");
        File sourceFile = new File(hubMinerSourceDir, pathTail);
        if (!sourceFile.exists()) {
            throw new Exception("File does not exist:" +
                    sourceFile.getPath());
        }
        UploadImplementation ui = client.openmlImplementationUpload(
                implementationFile, null, sourceFile);
        return ui.getId();
    }
    
    /**
     * This method creates an implementation object with the specified parameter
     * descriptions.
     * 
     * @param classifierClass Class that is the class of the classifier.
     * @param parameterDescriptions HashMap<String, String> mapping parameter
     * names to parameter descriptions.
     * @return Implementation that corresponds to the specified classifier class
     * and the specified list of parameters.
     * @throws Exception 
     */
    public static Implementation create(Class classifierClass,
            HashMap<String, String> parameterDescriptions) throws Exception {
        if (classifierClass == null) {
            return null;
        }
        ValidateableInterface classifier = (ValidateableInterface) (
                classifierClass.newInstance());
        String version = Long.toString(classifier.getVersion());
        String classPath = classifierClass.getName();
        String classifierName = classPath.substring(
                classPath.lastIndexOf('.') + 1);
        String description = "HubMiner implementation of " + classifierName;
        Implementation imp = new Implementation(classifierName, version,
                description, "java", " ");
        if (parameterDescriptions != null) {
            Set<String> paramNames = parameterDescriptions.keySet();
            for (String paramName: paramNames) {
                String paramDescription = parameterDescriptions.get(paramName);
                Field fld = classifierClass.getDeclaredField(paramName);
                fld.setAccessible(true);
                String paramType = fld.getType().toString();
                String defaultValue = fld.get(classifier).toString();
                imp.addParameter(paramType, paramType, defaultValue,
                        paramDescription);
            }
        }
        // Now add the special "external_preprocessing" parameter that is going
        // to be set by the experimental framework itself.
        String envParamName = "external_preprocessing";
        // This may be changes to set a formal type, but for now it is just a
        // concatenated description of pre-processing steps from the
        // experimental environment.
        String envParamType = "String";
        String envDefaultValue = "feature_noise_rate:0, label_noise_rate:0, "
                + "noise_type:uniform instance_selection:NONE, "
                + "proto_hubness:UNBIASED, app_knn_sets:FALSE";
        String envParamDescription = "This parameter is actually a list of the"
                + "parameters of the experimental framework that affect the"
                + "performance of algorithms that are being tested. Since they"
                + "greatly affect the results, they are reported in order to"
                + "distinguish between different result uploads and make"
                + "smarter comparisons.";
        imp.addParameter(envParamName, envParamType, envDefaultValue,
                envParamDescription);
        return imp;
    }
    
    /**
     * This method gets the list of current parameter values restricted on the
     * definitions in the implementation object.
     * 
     * @param paramValuesMap HashMap<String, Object> mapping parameter names to
     * their current values.
     * @param imp Implementation that holds the parametrization definitions.
     * @return ArrayList<Parameter_setting> that contains the current parameter
     * values.
     */
    public static ArrayList<Parameter_setting> getParameterSetting(
            HashMap<String, Object> paramValuesMap, Implementation imp) {
        ArrayList<Parameter_setting> settings = new ArrayList<>();
        for(Parameter p : imp.getParameter()) {
            try {
                String paramName = p.getName();
                if (paramValuesMap.containsKey(paramName)) {
                    String paramValue =
                            paramValuesMap.get(paramName).toString();
                    settings.add(new Parameter_setting(imp.getId(),
                            p.getName(), paramValue));
                }
            } catch (Exception e) {
                // Parameter not found.
            }
        }
        return settings;
    }
    
}
