package main.controller;

import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import javafx.stage.FileChooser.ExtensionFilter;
import javafx.stage.Stage;
import main.App;

import javax.annotation.PostConstruct;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class BrowserController {

    private static ExtensionFilter textFilter = new ExtensionFilter("Text files", "*.txt");
    private static ExtensionFilter imageFilter = new ExtensionFilter("Image files", "*.jpg", "*.png");
    private static ExtensionFilter xmlFilter = new ExtensionFilter("XML file", "*.xml");

    @FXML
    public Button browseImageBtn;
    @FXML
    public TextField outputFileName;
    @FXML
    public Button browseTextBtn;

    private Stage ownerStage;
    private File outputFolder;

    @PostConstruct
    public void initialize() {
        browseTextBtn.setOnAction(event -> openXmlFileBrowser());
        browseImageBtn.setOnAction(event -> openImageFileBrowser());
    }

    public void openXmlFileBrowser() {
        DirectoryChooser directoryChooser = new DirectoryChooser();
        File selectedDirectory = directoryChooser.showDialog(ownerStage);
        if (selectedDirectory != null) {
            outputFolder = selectedDirectory;
            outputFileName.setText(outputFolder.toString());
        }
    }

    private void openImageFileBrowser() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().setAll(imageFilter);
        List<File> files = fileChooser.showOpenMultipleDialog(ownerStage);
        if (files != null) {
            files.forEach(this::openViewerStage);
        }
    }

    private void openViewerStage(File file) {
        try {

            Image image = new Image(file.toURI().toURL().toExternalForm());
            FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("../view/viewer.fxml"));
            Parent loader = fxmlLoader.load();

            String nameTxt = file.getName();
            nameTxt = nameTxt.substring(0, nameTxt.lastIndexOf('.')) + ".txt";
            String txt = null;
            try {
                txt = new String(Files.readAllBytes(Paths.get(file.getParent() + "\\" + nameTxt)));
            } catch (Exception e) {
                Alert alert = new Alert(Alert.AlertType.ERROR);
                alert.setTitle("File not found");
                alert.setContentText("Can not find the text file corresponding to the chosen image (format file name: image_name_without_extension + \".txt\"");
                alert.showAndWait();
                return;
            }

            ViewerController viewerController = fxmlLoader.getController();
            viewerController.setOriginalImage(image, file.getName());
            viewerController.setBrowserController(this);
            viewerController.createXmlFileWriter(file.getName(), (int) image.getWidth(), (int) image.getHeight());
            viewerController.setGroundTruthTxt(txt);

            Stage viewerStage = new Stage();
            viewerStage.setTitle("Viewer");
            viewerStage.setMinHeight(300);
            viewerStage.setMinWidth(400);
            viewerStage.setScene(new Scene(loader, 800, 600));
            viewerStage.show();

        } catch (Exception e) {
            App.showExceptionAlert(e);
        }
    }

    public File getOutputFolder() {
        return outputFolder;
    }

    public void setOwnerStage(Stage ownerStage) {
        this.ownerStage = ownerStage;
    }
}