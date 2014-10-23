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
import learning.supervised.evaluation.ValidateableInterface;
import org.openml.apiconnector.algorithms.Conversion;
import org.openml.apiconnector.io.ApiSessionHash;
import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.apiconnector.xml.Implementation;
import org.openml.apiconnector.xml.ImplementationExists;
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
     * @param hashFetcher ApiSessionHash object for obtaining the session hash.
     * @return Integer that is the implementation ID obtained from the OpenML
     * servers.
     * @throws Exception 
     */
    public static int getImplementationId(Implementation implementation,
            Class classifierClass, File hubMinerSourceDir,
            OpenmlConnector client, ApiSessionHash hashFetcher)
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
                    hubMinerSourceDir, client, hashFetcher);
        }
        if(result != null && result.exists()) {
            return result.getId();
        } else {
            return registerImplementation(implementation, classifierClass,
                    hubMinerSourceDir, client, hashFetcher);
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
     * @param hashFetcher ApiSessionHash object for obtaining the session hash.
     * @return Integer that is the implementation ID obtained from the OpenML
     * servers.
     * @throws Exception 
     */
    private static int registerImplementation(Implementation implementation,
            Class classifierClass, File hubMinerSourceDir,
            OpenmlConnector client, ApiSessionHash hashFetcher)
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
                implementationFile, null, sourceFile,
                hashFetcher.getSessionHash());
        return ui.getId();
    }
    
    
    
}
