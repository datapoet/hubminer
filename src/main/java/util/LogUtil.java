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

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

/**
 * Utility class for logs to existing files. New entries are appended.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LogUtil {

    /**
     * @param logPath Path to an existing log file.
     * @return A writer object that appends new entries.
     * @throws Exception
     */
    public static PrintWriter getWriterToLogPath(String logPath)
            throws Exception {
        return new PrintWriter(new FileWriter(logPath, true), true);
    }

    /**
     * @param f An existing log file.
     * @return A writer object that appends new entries.
     * @throws Exception
     */
    public static PrintWriter getWriterToLog(File f) throws Exception {
        return new PrintWriter(new FileWriter(f.getPath(), true), true);
    }
}
