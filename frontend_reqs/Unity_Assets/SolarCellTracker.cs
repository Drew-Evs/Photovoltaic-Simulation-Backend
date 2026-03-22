using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Text;

public class SolarCellTracker : MonoBehaviour
{
    //the light source using
    [Header("Lighting")]
    public Transform light;

    //angular size in degrees
    [Header("Simulation Setting")]
    public float angularSize = 2.0f;

    //number of rays each cell sends - more is longer but smoother penumbra
    public int raysPerCell = 16;

    //difference between full sun and full shade
    public float maxIrr = 1000f;
    public float minIrr = 100f;

    //storing a list of cells
    [Header("Panel Config")]
    public List<GameObject> cells = new List <GameObject>();
    public float[] irrArray;

    //script to generate the cell grid
    [Header("Grid Generation")]
    //structure of grid
    public int rows = 8;
    public int columns = 6;
    //spacing of cells and their size
    public float cellSize = 0.156f;
    public float spacing = 0.005f;

    //settings to connect to the python backend
    [Header("Network Settings")]
    public string pythonIP = "127.0.0.1";
    public int port = 5005;
    private UdpClient udpClient;

    //shows that the light controller is waiting for python
    [Header("Lockstep Components")]
    public LightController lightController;
    private bool isWaitingForPython = false;

    //clickable button to create it 
    [ContextMenu("Generate Grid")]
    public void GenerateGrid() {
        //delete to reset
        foreach (GameObject cell in cells) {
            if (cell != null) DestroyImmediate(cell);
        }
        cells.Clear();

        //build new grid
        for (int r=0; r<rows; r++) {
            for (int c=0; c<columns; c++) {
                //create cell as a quad (3D flat square)
                GameObject newCell = GameObject.CreatePrimitive(PrimitiveType.Quad);
                newCell.name = $"Cell_{r}_{c}";

                //group to solar panel object
                newCell.transform.SetParent(this.transform);

                //position in the grid layout and sclae to correct size
                float xPos = c * (cellSize + spacing);
                float yPos = r * (cellSize + spacing); 
                newCell.transform.localPosition = new Vector3(xPos, yPos, 0);
                newCell.transform.localScale = new Vector3(cellSize, cellSize, 1);

                cells.Add(newCell);
            }
        }
    }

    void Start() {
        //a standard array
        irrArray = new float[cells.Count];

        //need to initate network client and send data function
        udpClient = new UdpClient();
    }

    void Update() {
        //check if waiting for python - halt if it is
        if (isWaitingForPython) {
            CheckForPythonReply();
            return;
        }

        //get the base direction to light source and rotation basis to generate a cone
        Vector3 directionToLight = -light.forward;
        //used for safe rotation
        Quaternion lightRotation = Quaternion.LookRotation(directionToLight);

        //iterate through each cell, moving the start of the ray to the new cell
        for (int i=0; i<cells.Count; i++) {
            GameObject cell = cells[i];

            //need the cell normal (forward direction)
            Vector3 cellNormal = -cell.transform.forward;
            Vector3 rayStart = cell.transform.position + (cellNormal * 0.01f);

            //want to calculate angle of incidence to map effective irradiance (using formula irr = irr_max * cos(theta))
            float angleCosine = Vector3.Dot(cellNormal, directionToLight);
            //ensure positive
            angleCosine = Mathf.Clamp01(angleCosine);
            //and get max effective
            float effectiveMaxIrr = maxIrr * angleCosine;

            //track number of hits to decide how shaded
            int hits = 0;

            //iterate through a scattering of rays (in a cone shape)
            //reference Monte Carlo Raycasting
            for (int r=0; r<raysPerCell; r++) {
                //random point from a 2D circle and convert to a slight angle offset (make a cone)
                Vector2 randomPoint = UnityEngine.Random.insideUnitCircle;
                float angleX = randomPoint.x * (angularSize / 2f);
                float angleY = randomPoint.y * (angularSize / 2f);

                //apply this offset to the vector
                Quaternion rayDeviation = Quaternion.Euler(angleX, angleY, 0);
                Vector3 rayDirection = lightRotation * rayDeviation * Vector3.forward;

                //random point on the cell to shoot from + apply offset
                float randomX = UnityEngine.Random.Range(-cellSize / 2f, cellSize / 2f);
                float randomY = UnityEngine.Random.Range(-cellSize / 2f, cellSize / 2f);
                Vector3 randomizedRayStart = rayStart + (cell.transform.right * randomX) + (cell.transform.up * randomY);

                //then fire and check if it doesnt collide (successfully hit the light)
                if (!Physics.Raycast(randomizedRayStart, rayDirection, Mathf.Infinity)) {
                    hits++;
                    //draw the line 
                    Debug.DrawRay(randomizedRayStart, rayDirection * 5f, Color.green);
                } else {
                    Debug.DrawRay(randomizedRayStart, rayDirection * 5f, Color.red);
                }
            }
            //calculate percentage of light visible on the cell 
            float lightVisibility = (float)hits / raysPerCell;

            //interpolate to get the irradiance and base a colour based off that
            irrArray[i] = Mathf.Lerp(0, effectiveMaxIrr, lightVisibility);
            Color cellColor = Color.Lerp(Color.blue, Color.yellow, lightVisibility * angleCosine);
            cell.GetComponent<MeshRenderer>().material.color = cellColor;
        }

        //update to python
        SendIrrPython();
        isWaitingForPython = true;
    }

    //function to package and send the array
    void SendIrrPython() {
        //dont want to send empty
        if (irrArray == null || irrArray.Length == 0) return;

        //convert to comma seperated string to send
        string dataString = string.Join(",", irrArray);
        byte[] dataBytes = Encoding.UTF8.GetBytes(dataString);

        //then fire
        udpClient.Send(dataBytes, dataBytes.Length, pythonIP, port);
    }

    //clear client when stopped
    void OnApplicationQuit() {
        if (udpClient != null) {
            udpClient.Close();
        }
    }

    //check for a python update
    void CheckForPythonReply() {
        //checks if packet waiting 
        if (udpClient.Available > 0) {
            System.Net.IPEndPoint remoteEP = new System.Net.IPEndPoint(System.Net.IPAddress.Any, 0);
            byte[] data = udpClient.Receive(ref remoteEP);
            string reply = Encoding.UTF8.GetString(data);


            //move the light one step
            if (lightController != null) {
                lightController.StepLight();
            }

            //unpause for next frame
            isWaitingForPython = false;
        }
    }
}
