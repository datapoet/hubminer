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
package learning.supervised.evaluation;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This class lists and sets the parameter values for classification algorithms.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClassifierParametrization {
    
    /**
     * This method gets a HashMap containing the values of parameters used in
     * a particular classifier.
     * 
     * @param classifier ValidateableInterface classifier to get the values for.
     * @return HashMap<String, Object> representing the classifier parameter
     * values.
     */
    public static HashMap<String, Object> getClassifierParameterValues(
            ValidateableInterface classifier) {
        Class clClass = classifier.getClass();
        HashMap<String, String> paramNameDescMap =
                classifier.getParameterNamesAndDescriptions();
        Set<String> paramNames = paramNameDescMap.keySet();
        HashMap<String, Object> paramNameValueMap =
                new HashMap<>(paramNames.size());
        for (String paramName: paramNames) {
            try {
                Field fld = clClass.getDeclaredField(paramName);
                fld.setAccessible(true);
                Object fieldValue = fld.get(classifier);
                paramNameValueMap.put(paramName, fieldValue);
            } catch (NoSuchFieldException | SecurityException |
                    IllegalAccessException e) {
                System.err.println(e.getMessage());
            }
        }
        // Handle the metrics, as they are not explicitly specified by
        // individual classifier objects in parameter lists, since they are the
        // parameter of the base classifier class.
        try {
            Field fld = clClass.getSuperclass().getDeclaredField("cmet");
            fld.setAccessible(true);
            Object fieldValue = fld.get(classifier);
            paramNameValueMap.put("cmet", fieldValue);
        } catch (NoSuchFieldException | SecurityException |
                    IllegalAccessException e) {
            System.err.println(e.getMessage());
        }
        return paramNameValueMap;
    }
    
    /**
     * This method gets a HashMap containing the values of parameters used in
     * a particular classifier, as strings.
     * 
     * @param classifier ValidateableInterface classifier to get the values for.
     * @return HashMap<String, Object> representing the classifier parameter
     * values.
     */
    public static HashMap<String, String> getClassifierParameterStringValues(
            ValidateableInterface classifier) {
        HashMap<String, Object> paramValueMap =
                getClassifierParameterValues(classifier);
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
     * This method sets the provided parameter values to a classifier. It throws
     * an Exception if a wrong parameter name is provided, which does not exist
     * in the implementation.
     * 
     * @param classifier ValidateableInterface that is the classifier to set the
     * parameter values to.
     * @param paramValuesMap HashMap<String, Object> that maps the parameter
     * values to set to the classifier.
     */
    public static void setParameterValuesToClassifier(
            ValidateableInterface classifier,
            HashMap<String, Object> paramValuesMap)
            throws NoSuchFieldException, IllegalArgumentException,
            IllegalAccessException {
        if (classifier == null || paramValuesMap == null) {
            return;
        }
        Class clClass = classifier.getClass();
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
                        fld.set(classifier, fValue);
                    } else if (fld.getType().equals(Integer.TYPE) &&
                            (paramValue.getClass().equals(
                            Class.forName("java.lang.Double")) ||
                            paramValue.getClass().equals(
                            Class.forName("java.lang.Float")))) {
                        int iValue = Math.round(
                                ((Double) paramValue).floatValue());
                        fld.set(classifier, iValue);
                    } else {
                        fld.set(classifier, paramValue);
                    }
                } catch (ClassNotFoundException ex) {
                    Logger.getLogger(
                            ClassifierParametrization.class.getName()).log(
                            Level.SEVERE, null, ex);
                }
            }
        }
    }
}
