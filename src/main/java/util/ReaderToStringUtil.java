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
package util;

import java.io.Reader;

/**
 * This class implements a method that takes a reader and converts its contents
 * into a single String.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ReaderToStringUtil {
    
    /**
     * This method reader a Reader as a single string.
     * 
     * @param reader Reader object to be read as single String.
     * @return String representing the contents of the Reader.
     * @throws Exception 
     */
    public static String readAsSingleString(Reader reader) throws Exception {
        if (reader == null) {
            return "";
        }
        int bufferLength = 524288;
        char[] charArray = new char[bufferLength];
        StringBuilder builder = new StringBuilder();
        int numReadCharacters;
        while ((numReadCharacters =
                reader.read(charArray, 0, bufferLength)) > 0) {
            builder.append(charArray, 0, numReadCharacters);
        }
        return builder.toString();
    }
}
