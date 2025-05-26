import React, { useEffect, useState, useRef, act } from 'react';
import io from 'socket.io-client';
import './App.css'
import infernoLogo from '/inferno.jpg'


const usb_cam_server = 'http://192.168.31.91:5000/' ;
const ip_cam_server = 'http://localhost:3001/' ;

const socket = io('http://localhost:3000') ;  


socket.on('new-connection' ,(client)=>{
  let msg = `new ${client} connected to server`;
  alert(msg);
})

function App() {

  function handleKill(){
    socket.emit('kill', "t");
  }
  
  return (
    <>
      <div className="head">
        <img src="/logo.png" height='60px' alt="" />
        <button className='kill-switch' onClick={handleKill}>KILL</button>
      </div>


      <VideoFeed />
    </>
  )
}




const VideoFeed = () => {
  
  const [active, setAcitve] = useState(1);

  const videoArray = [
    { id: 1, name: 'cam1', imgSrc: usb_cam_server+'video_feed/1' },
    { id: 2, name: 'cam2', imgSrc: usb_cam_server+'video_feed/2' },
    { id: 3, name: 'cam3', imgSrc: infernoLogo },
    { id: 4, name: 'cam4', imgSrc: infernoLogo },
    { id: 5, name: 'cam5', imgSrc: infernoLogo },
    { id: 6, name: 'cam6', imgSrc: infernoLogo },
  ];
  
  return (
    <div className='videos'>
      {videoArray.map((vdo) => {
        return (
          <div
          className={(active === vdo.id) ? "vdo-wrap active" : "vdo-wrap"}
            key={vdo.id}
          >
            <img
              src={vdo.imgSrc}
              alt=""
              onClick={() => setAcitve(vdo.id)}

              />
            <p>{vdo.name}</p>
          </div>
        )
      })
    }
      <DriveDiv socket={socket}/>
      <ArmDiv socket={socket}/>
    </div>
  );
};




function DriveDiv({socket}) {
  const [messages, setMessages] = useState([]);


  useEffect(() => {
    socket.emit('new-connection', 'front-end-client');

    socket.on('drive-client', (msg) => {
      setMessages((prevMessages) => {
        let newArray = [...prevMessages, msg].slice(-5);
        console.log(newArray);
        return newArray;
      });
    });

    return () => {
      socket.off('drive-client');
    };
  }, []);

  return (
    <div className="data drive-data">
      {
        messages.map((msg, i) => {
          return (
            <p key={i}>{msg}</p>
          )
        })
      }
    </div>
  )
}


function ArmDiv({socket}) {
  const [messages, setMessages] = useState([]);


  useEffect(() => {
    socket.emit('front-end-connection', 'front-end client connected');

    socket.on('arm-client', (msg) => {
      setMessages((prevMessages) => {
        let newArray = [...prevMessages, msg].slice(-5);
        console.log(newArray);
        return newArray;
      });
    });

    return () => {
      socket.off('arm-client');
    };
  }, []);

  return (
    <div className="data arm-data">
      {
        messages.map((msg, i) => {
          return (
            <p key={i}>{msg}</p>
          )
        })
      }
    </div>
  )
}




export default App
