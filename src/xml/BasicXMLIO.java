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
package xml;

import ioformat.FileUtil;
import java.io.File;
import java.io.FileOutputStream;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import org.w3c.dom.Document;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

/**
 * Basic XML I/O operations.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BasicXMLIO {

    /**
     * Get Document object from an XML in the specified file.
     *
     * @param f File that stores the XML.
     * @return Document object representing the XML in the file.
     * @throws Exception
     */
    public static Document getDocumentFromXML(File f) throws Exception {
        if (f == null || !(f.exists() && f.isFile())) {
            throw new Exception("Invalid file path provided.");
        }
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document document = builder.parse(f.getPath());
        return document;
    }

    /**
     * Write the document object to a file in XML format.
     *
     * @param document Document object to persist.
     * @param f File to write the document to.
     * @throws Exception
     */
    public static void writeDocumentToXML(Document document,
            File f) throws Exception {
        FileUtil.createFile(f);
        TransformerFactory transFactory = TransformerFactory.newInstance();
        Transformer transformer = transFactory.newTransformer();
        DOMSource source = new DOMSource(document);
        FileOutputStream os = new FileOutputStream(f);
        StreamResult result = new StreamResult(os);
        transformer.transform(source, result);
    }
}
