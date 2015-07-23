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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

/**
 * A utility class for some simple HTTP requests.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HTTPUtil {
    
    public static final String DEFAULT_USER_AGENT = "Chrome";
    
    /**
     * Make a GET HTTP request.
     * 
     * @param urlString String that represents the URL.
     * @return String that is the response.
     * @throws Exception 
     */
    public static String get(String urlString) throws Exception {
        return get(urlString, DEFAULT_USER_AGENT);
    }
    
    /**
     * Make a GET HTTP request.
     * 
     * @param urlString String that represents the URL.
     * @param userAgent String representing the user agent.
     * @return String that is the response.
     * @throws Exception 
     */
    public static String get(String urlString, String userAgent)
            throws Exception {
        URL urlObject = new URL(urlString);
        HttpURLConnection conn = (HttpURLConnection) urlObject.openConnection();
        conn.setRequestMethod("GET");
        conn.setRequestProperty("User-Agent", userAgent);
        StringBuilder receivedContent;
        try (BufferedReader br = new BufferedReader(
                     new InputStreamReader(conn.getInputStream()))) {
            receivedContent = new StringBuilder();
            String line = br.readLine();
            while (line != null) {
                receivedContent.append("\n");
                receivedContent.append(line);
                line = br.readLine();
            }
        }
        String contentString = receivedContent.toString();
        return contentString;
    }
    
    /**
     * Make a POST HTTP request.
     * 
     * @param urlString String that represents the URL.
     * @param userAgent String representing the user agent.
     * @param postString String that is being POST-ed.
     * @return String that is the response.
     * @throws Exception 
     */
    public static String post(String urlString, String userAgent,
            String contentString, String contentType)
            throws MalformedURLException, IOException {
        byte[] contentBytes = contentString.getBytes();
        URL urlObject = new URL(urlString);
        HttpURLConnection conn = (HttpURLConnection) urlObject.openConnection();
        conn.setDoOutput(true);
        conn.setRequestMethod("POST");
        // For instance, content type can be something like:
        // "application/json; charset=utf8".
        conn.setRequestProperty("Content-Type", contentType);
        conn.setRequestProperty("Content-Length",
                String.valueOf(contentBytes.length));
        conn.setRequestProperty("User-Agent", userAgent);
        conn.connect();
        try (OutputStream os = conn.getOutputStream()) {
            os.write(contentString.getBytes());
            os.flush();
        }
        InputStream response = conn.getInputStream();
        InputStreamReader reader = new InputStreamReader(response);
        String resultString;
        try (BufferedReader br = new BufferedReader(reader)) {
            resultString = br.readLine();
        }
        return resultString;
    }
}
