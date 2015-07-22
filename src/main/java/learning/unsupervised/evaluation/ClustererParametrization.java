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
package learning.unsupervised.evaluation;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import learning.unsupervised.ClusteringAlg;

/**
 * This class lists and sets the parameter values for clustering algorithms.
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClustererParametrization {
    /**
     * This method gets a HashMap containing the values of parameters used in
     * a particular clustering algorithm.
     * 
     * @param clusterer ClusteringAlg to get the values for.
     * @return HashMap<String, Object> representing the classifier parameter
     * values.
     */
    public static HashMap<String, Object> getClustererParameterValues(
            ClusteringAlg clusterer) {
        Class clClass = clusterer.getClass();
        HashMap<String, String> paramNameDescMap =
                clusterer.getParameterNamesAndDescriptions();
        Set<String> paramNames = paramNameDescMap.keySet();
        HashMap<String, Object> paramNameValueMap =
                new HashMap<>(paramNames.size());
        for (String paramName: paramNames) {
            try {
                Field fld = clClass.getDeclaredField(paramName);
                fld.setAccessible(true);
                Object fieldValue = fld.get(clusterer);
                paramNameValueMap.put(paramName, fieldValue);
            } catch (NoSuchFieldException | SecurityException |
                    IllegalAccessException e) {
                System.err.println(e.getMessage());
            }
        }
        // Handle the metrics, as they are not explicitly specified by
        // individual clusterer objects in parameter lists, since they are the
        // parameter of the base clusterer class.
        try {
            Field fld = clClass.getSuperclass().getDeclaredField("cmet");
            fld.setAccessible(true);
            Object fieldValue = fld.get(clusterer);
            paramNameValueMap.put("cmet", fieldValue);
        } catch (NoSuchFieldException | SecurityException |
                    IllegalAccessException e) {
            System.err.println(e.getMessage());
        }
        return paramNameValueMap;
    }
    
    /**
     * This method gets a HashMap containing the values of parameters used in
     * a particular clusterer, as strings.
     * 
     * @param clusterer ClusteringAlg to get the values for.
     * @return HashMap<String, Object> representing the clusterer parameter
     * values.
     */
    public static HashMap<String, String> getClustererParameterStringValues(
            ClusteringAlg clusterer) {
        HashMap<String, Object> paramValueMap =
                getClustererParameterValues(clusterer);
        HashMap<String, String> paramStringValueMap = new HashMap<>();
        if (paramValueMap == null || paramValueMap.isEmpty()) {
            return paramStringValueMap;
        }
        Set<String> paramNames = paramValueMap.keySet();
        for (String paramName: paramNames) {
            paramStringValueMap.put(paramName,
                    paramValueMap.get(paramName).toString());
        }
        return paramStringValueMap;
    }
    
    /**
     * This method sets the provided parameter values to a clusterer. It throws
     * an Exception if a wrong parameter name is provided, which does not exist
     * in the implementation.
     * 
     * @param clusterer ClusteringAlg to get the values for.
     * @param paramValuesMap HashMap<String, Object> that maps the parameter
     * values to set to the clusterer.
     */
    public static void setParameterValuesToClusterer(
            ClusteringAlg clusterer,
            HashMap<String, Object> paramValuesMap)
            throws NoSuchFieldException, IllegalArgumentException,
            IllegalAccessException {
        if (clusterer == null || paramValuesMap == null) {
            return;
        }
        Class clClass = clusterer.getClass();
        Set<String> paramNames = paramValuesMap.keySet();
        for (String paramName: paramNames) {
            Object paramValue = paramValuesMap.get(paramName);
            Field fld = clClass.getDeclaredField(paramName);
            fld.setAccessible(true);
            if (paramValue != null) {
                try {
                    if (fld.getType().equals(Float.TYPE) &&
                            paramValue.getClass().equals(
                            Class.forName("java.lang.Double"))) {
                        float fValue = ((Double) paramValue).floatValue();
                        fld.set(clusterer, fValue);
                    } else if (fld.getType().equals(Integer.TYPE) &&
                            (paramValue.getClass().equals(
                            Class.forName("java.lang.Double")) ||
                            paramValue.getClass().equals(
                            Class.forName("java.lang.Float")))) {
                        int iValue = Math.round(
                                ((Double) paramValue).floatValue());
                        fld.set(clusterer, iValue);
                    } else {
                        fld.set(clusterer, paramValue);
                    }
                } catch (ClassNotFoundException ex) {
                    Logger.getLogger(
                            ClustererParametrization.class.getName()).log(
                            Level.SEVERE, null, ex);
                }
            }
        }
    }
}
