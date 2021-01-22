# Hide and Seek
Teach two Waveshare JetBots how to play hide and seek in a closed arena.

## Instructions
1. Make sure that the IP address of the server in hider_QL.py and seeker_QL.py is that of the computer that will be used as the central control unit.
2. Set the number of episodes and iterations in each file (they must be equal in all three files).
3. Run <code>$ python3 jetbot_server.py</code> from the central control unit.
4. Run <code>$ python3 seeker_QL.py</code> from the robot that will play the Seeker role.
5. Once the Seeker is connected to the server, a message will appear on the screen. Then, run <code>$ python3 hider_QL.py</code> from the robot that will play the Seeker role.
6. Follow the instructions on the screen to train the robots.
