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

Welcome to Hub Miner!

<img src="HubMinerLogo.jpg" alt="Hub Miner logo" height="192" width="634">

It is a machine learning library aimed specifically at overcoming issues in high-dimensional data analysis and is focused mostly at the phenomenon of hubness, which is the asymmetric distribution of relevance within the models. It is a detrimental aspect of the well known curse of dimensionality, so hubness-aware methods (implicit or explicit) are necessary for effective instance-based learning in many dimensions. Hub Miner implements custom methods for classification, clustering, metric learning, instance selection and other common machine learning tasks, as well as a set of baselines and a powerful experimental framework that includes testing under various challenging conditions. Common data file formats are supported, including ARFF, csv and tsv. A full manual is in preparation, but the code is very well documented, so You are encouraged to have a look at some of the source files and online resources at http://ailab.ijs.si/nenad_tomasev/hub-miner-library/ and contact the author with any questions at this stage. The code is tested, stable and reliable - or so it should be - so feel free to notify of any particular issues if they arise.

This is the first release and updates are already under way, so - expect this library to grow and be even better documented and supported.

As for dependencies, these are is the current list:

apiconnector-fat.jar
collections-generic-4.01.jar
colt-1.2.0.jar
commons-codec-1.3.jar
commons-httpclient-3.0.1.jar
commons-logging-1.1.jar
concurrent-1.3.4.jar
gson-2.3.jar
iText-2.1.7_mx-1.0.jar
Jama-1.0.2.jar
jcommon-1.0.17.jar
jdom.jar
jetty-6.1.1.jar
jetty-util-6.1.1.jar
jfreechart-1.0.14.jar
jgraph.jar
jgraphx.jar
json.jar
jsoup-1.7.2.jar
jtidy-r7.jar
jung-algorithms-2.0-beta1.jar
jung-api-2.0-beta1.jar
jung-graph-impl-2.0-beta1.jar
jung-jai-samples-2.0-beta1.jar
jung-visualization-2.0-beta1.jar
junit-4.7.jar
mdsj.jar
mxgraph-all.jar
rome-0.8.jar
servlet-api-2.5-6.1.1.jar
servlet.jar
swing-layout-1.0.3.jar
swingx-1.6.jar
swingx-beaninfo-1.6.jar
swingx-ws-1.0.jar
TGGraphLayout.jar
xercesImpl.jar
xmlunit1.0.jar

A dependency on OpenML is apiconnector-fat.jar and it can be downloaded from http://openml.org/downloads/apiconnector-fat.jar

All Hub Miner code is in Java and should be portable.

A small part of library that has to do with SIFT feature analysis still relies on having the SiftWin binary in the path and ImageMagick. However, this is just a few methods and unless You plan to use Hub Miner for image feature extraction (which is not its main purpose) - You should be fine without it. I intend to remove this dependency in future builds and switch over to some Java-based image feature extraction libraries, as well as provide better support for OpenCV formats.