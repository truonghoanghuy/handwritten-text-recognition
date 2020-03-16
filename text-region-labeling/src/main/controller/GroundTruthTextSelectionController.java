package main.controller;

import javafx.embed.swing.SwingFXUtils;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.KeyCode;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Modality;
import javafx.util.Pair;
import main.App;
import main.controller.BrowserController;
import main.utils.XmlFileWriter;

import javax.annotation.PostConstruct;
import java.awt.*;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Iterator;
import java.util.LinkedList;

public class GroundTruthTextSelectionController {
    @FXML
    private Button okBtn;
    @FXML
    private Button cancelBtn;
    @FXML
    private TextArea paragraphTxt;
    @FXML
    private TextArea selectedTxt;
    @FXML
    private ScrollPane scrollPane1;
    @FXML
    private ScrollPane scrollPane2;

    private StringBuilder outputString;
    private StringBuilder groundTruth;

    @PostConstruct
    public void initialize() {
        paragraphTxt.prefHeightProperty().bind(scrollPane1.heightProperty());
        paragraphTxt.prefWidthProperty().bind(scrollPane1.widthProperty());
        selectedTxt.prefHeightProperty().bind(scrollPane2.heightProperty());
        selectedTxt.prefWidthProperty().bind(scrollPane2.widthProperty());

        paragraphTxt.setOnMouseClicked(this::handleTextSelection);
    }

    private void handleTextSelection(MouseEvent event) {
        selectedTxt.setText(paragraphTxt.getSelectedText());
    }

    public void setProperty(StringBuilder output, StringBuilder groundTruthTxt) {
        this.outputString = output;
        this.groundTruth = groundTruthTxt;
        paragraphTxt.setText(this.groundTruth.toString());
    }
}