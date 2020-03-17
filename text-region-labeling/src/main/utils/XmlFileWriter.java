package main.utils;

import java.io.File;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.w3c.dom.Document;
import org.w3c.dom.Element;

public class XmlFileWriter {
    private Document doc;
    private Element ns;
    private Element page;
    private Element curTextRegion;

    private int curRegionId = 1;
    private int curLineId = 1;

    public XmlFileWriter(String fileName, int imageWidth, int imageHeight) {
        try {
            DocumentBuilderFactory docFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder docBuilder = docFactory.newDocumentBuilder();
            doc = docBuilder.newDocument();
            doc.setXmlStandalone(true);
        } catch (Exception e) {
            e.printStackTrace();
        }

        ns = doc.createElementNS("http://www.cse.hcmut.edu.vn/", "thesis");
        doc.appendChild(ns);

        page = doc.createElement("Page");
        page.setAttribute("imageFilename", fileName);
        page.setAttribute("imageWidth", imageWidth + "");
        page.setAttribute("imageHeight", imageHeight + "");
        ns.appendChild(page);
    }

    public void addTextRegion(String points) {
        Element textRegion = doc.createElement("TextRegion");
        textRegion.setAttribute("id", "pg_" + curRegionId++);

        Element coords = doc.createElement("Coords");
        coords.setAttribute("points", points);
        textRegion.appendChild(coords);

        page.appendChild(textRegion);
        curTextRegion = textRegion;
    }

    public void addTextLine(String boundary, String baseLinePoints, String groundTruth) {
        Element textLine = doc.createElement("TextLine");
        textLine.setAttribute("id", "line_" + curLineId++);

        Element coords = doc.createElement("Coords");
        coords.setAttribute("points", boundary);
        textLine.appendChild(coords);

        Element baseLine = doc.createElement("Baseline");
        baseLine.setAttribute("points", baseLinePoints);
        textLine.appendChild(baseLine);

        Element textEquiv = doc.createElement("TextEquiv");
        Element unicode = doc.createElement("Unicode");
        unicode.appendChild(doc.createTextNode(groundTruth));
        textEquiv.appendChild(unicode);
        textLine.appendChild(textEquiv);

        curTextRegion.appendChild(textLine);
    }

    public void writeXmlFile(File file) {
        try {
            TransformerFactory transformerFactory = TransformerFactory.newInstance();
            Transformer transformer = transformerFactory.newTransformer();
            DOMSource source = new DOMSource(doc);
            StreamResult result = new StreamResult(file);
            transformer.setOutputProperty(OutputKeys.STANDALONE, "yes");
            transformer.transform(source, result);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}