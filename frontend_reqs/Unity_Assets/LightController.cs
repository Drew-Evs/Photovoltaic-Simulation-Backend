using UnityEngine;

public class LightController : MonoBehaviour
{
    [Header("Light Movement Settings")]
    [Tooltip("How fast light spins (degrees per simulation frame)")]
    public float stepSize = 1f;

    [Tooltip("The light rotation axis. X-axis simulates sunrise to sunset")]
    public Vector3 rotationAxis = Vector3.right;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
    }

    public void StepLight()
    {
        //rotate smoothly
        transform.Rotate(rotationAxis * stepSize);
    }
}
