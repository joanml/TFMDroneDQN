import time
import Agentes
from Agentes.Drone import Drone, Master, Slave
from Agentes.AgenteSpadeDrone import PeriodicSenderAgent, ReceiverAgent, Config

if __name__ == "__main__":
    jidMaster = "senderMasterAgent13@gtirouter.dsic.upv.es"
    jidSlave = "reciberSlaveAgent13@gtirouter.dsic.upv.es"

    MasterConfig = Config(jidMaster,"123","Drone_1","MyLidar1_1","MyLidar1_2","GPS_1",jidSlave)
    SlaveConfig = Config(jidMaster,"123","Drone_2","MyLidar2_1","MyLidar2_2","GPS_2")

    master = PeriodicSenderAgent(MasterConfig)
    slave = ReceiverAgent(SlaveConfig)
    slave.start()
    master.start()
    while slave.is_alive():
        try:
            time.sleep(1)
        except KeyboardInterrupt:

            slave.stop()
            master.stop()
            break
    print("Agents finished")

    # #senderagent.start()
    #
    # dron1 = Drone(n="Drone_1",L1="MyLidar1_1",L2="MyLidar1_2", GPS='GPS_1')
    # dron2 = Drone(n="Drone_2",L1="MyLidar2_1",L2="MyLidar2_2", GPS='GPS_2')
    #
    # #print(dron1.allInfo())
    # #print(dron2.allInfo())
    # dron1.takeoff()
    # dron2.takeoff()
    # dron1.moveZ(5, 10)
    # dron2.moveZ(10, 10)
    # position1 = dron1.getPosition()
    # position2 = dron2.getPosition()
    # if position1 == position2:
    #     print("Iguales")
    # else:
    #     print(position1,position2)
    # for i in range(15):
    #     dron1.getLidar1()
    #     dron1.getLidar2()
    #     position1 = dron1.getPosition()
    #     sleep(1)
    #     x=0
    #     y=5+i*2
    #     dron1.moveTo(10, -y, 15, 10)
    #     dron2.moveTo(position1.x_val,position1.y_val,-position1.z_val,10)
    #     sleep(1)

    #dron.move(10,20,10,10)
    #dron.moveTo(100, 0, -9, 5)
    #dron.getImage()



    # print(dron.allInfo())
    # dron.move(10,20)
    # dron.getImage()
    # print(dron.allInfo())


    ##############################################################################
    #     dron1 = DroneAgente.DroneAgente("drone@gtirouter.dsic.upv.es","GPS1")
    #     print(dron1.allInfo())
    #     dron1.takeoff()
    #     print(dron1.allInfo())
    #
    #     # nombre = "drone@gtirouter.dsic.upv.es", "123"
    #     # future = receiveragent.start()
    #     # future.result() # wait for receiver agent to be prepared.
    #     # senderagent = Agente.SenderAgent("sender@gtirouter.dsic.upv.es", "123")
    #     # senderagent.start()
    #     #
    #     # while receiveragent.is_alive():
    #     #     try:
    #     #         time.sleep(1)
    #     #     except KeyboardInterrupt:
    #     #         senderagent.stop()
    #     #         receiveragent.stop()
    #     #         break
    #     # print("Agents finished")