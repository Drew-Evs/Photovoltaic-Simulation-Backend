import socket
import time
import pandas as pd

def run_udp_server():
    #configure network
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005

    #create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    #sock.settimeout(2.0)

    print(f'Data Harvester listening on {UDP_IP}:{UDP_PORT}')
    print("Waiting for Unity to start sending data...")
    start_time = time.time()

    #harvest irr list as data - can solve quicker seperately
    rows = []

    try:
        #loop for a day
        for step in range(1500):
            # wati for unity packet
            data, addr = sock.recvfrom(4096)

            message = data.decode('utf-8')

            #convert to irradiance list
            irr_list = [float(x) for x in message.split(',')]

            elapsed_time = time.time() - start_time

            #handshake format
            reply_message = "0.00,DataSaved"
            sock.sendto(reply_message.encode('utf-8'), addr)
            print(f"[Step {step:04d} | {elapsed_time:.1f}s] | Harvested {len(irr_list)} cells")

            #save dynamically to own time
            row_data = {}

            for idx, irr_val in enumerate(irr_list):
                row_data[f"cell_{idx}"] = irr_val

            rows.append(row_data)

    except Exception as e:
        print(f"Error processing packet: {e}")

    #let write early
    except KeyboardInterrupt:
        print("\nSimulation manually stopped by user (Ctrl+C).")
    
    finally:
        #save dataset to csv and clean up
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv("Saved_Irradiance_Data.csv", index=False)
            print(f"\nSuccessfully saved {len(rows)} steps to 'Saved_Irradiance_Data.csv'")

        sock.close()

if __name__ == "__main__":
    run_udp_server()