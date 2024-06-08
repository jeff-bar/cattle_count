import React, { useEffect, useRef, useState } from 'react';
import Peer from 'simple-peer';
import './App.css'; // Importando o arquivo CSS para estilos

const App = () => {
  const ipServer = "ws://localhost:8000"

  const videoRef = useRef(null);
  const peerRef = useRef(null);

  const wsRefVideo = useRef(null);
  const wsRefCattleInfo = useRef(null);

  const [amount, setAmount] = useState(null);
  const [weight, setWeight] = useState(null);
  



  useEffect(() => {
    // Conexão WebSocket com o servidor
    const wsCattleInfo = new WebSocket(`${ipServer}/ws_info`);
    wsRefCattleInfo.current = wsCattleInfo;

    wsCattleInfo.onopen = () => {
      console.log('WebSocket connected Cattle Info');
    };

    wsCattleInfo.onmessage = async (event) => {
      const data = JSON.parse(event.data);

      console.log( data.type )

      if (data.type === 'cattle_info') {
        // Atualizar o número com o valor recebido do servidor
        setAmount(data.amount);
        setWeight(data.weight);
      }
    };

    wsCattleInfo.onerror = (error) => {
      console.error("WebSocket error Cattle Info: ", error);
    };

    wsCattleInfo.onclose = (event) => {
      console.log(`WebSocket closed with code Cattle Info: ${event.code}`);
    };

    return () => {
      wsCattleInfo.close();
    };
  }, []);


  useEffect(() => {
    const wsVideo = new WebSocket(`${ipServer}/ws`);
    wsRefVideo.current = wsVideo;

    wsVideo.onopen = () => {
      console.log('WebSocket connected Video');
    };

    wsVideo.onmessage = async (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'offer') {
        const peer = new Peer({ initiator: false, trickle: false });
        peerRef.current = peer;

        peer.on('signal', (signalData) => {
          if ('sdp' in signalData) { 
            wsVideo.send(JSON.stringify({ type: 'answer', sdp: signalData.sdp, sdp_type: signalData.type }));
          }
        });

        peer.on('stream', (stream) => {
          videoRef.current.srcObject = stream;
        });

        peer.signal({
          type: data.sdp_type,
          sdp: data.sdp
        });
      } else if (data.type === 'candidate' && peerRef.current !== null && peerRef.current !== undefined) {
        peerRef.current.signal(data.candidate);
      }
    };

    wsVideo.onerror = (error) => {
      console.error("WebSocket error Video : ", error);
    };

    wsVideo.onclose = (event) => {
      console.log(`WebSocket closed with code Video: ${event.code}`);
    };

    return () => {
      wsVideo.close();
    };
  }, []);



  return (
    <div className="app">
      <div className="video-container">

        <video 
            ref={videoRef} 
            autoPlay 
            muted 

            style={{ 
              width:'352px',
              //width: '100%', 
              height: 'auto', // Mantém a proporção original do vídeo
              maxWidth: '100%', 
              margin: '10px auto',
              borderRadius: '15px', // Define a borda arredondada
              display: 'block' // Centraliza o vídeo dentro do contêiner
            }}  />

        <div className="card-container">
          <div className="card">
            <h2>Quantidade</h2>
            <p>{amount !== null ? amount : 'Loading...'}</p> 
          </div>
          {/* 
          <div className="card">
            <h2>Pesagem</h2>
            <p>{weight !== null ? weight : 'Loading...'}</p> 
          </div>
        */}
        </div>
      </div>
    </div>
  );
};

export default App;
