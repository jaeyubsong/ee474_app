const { createProxyMiddleware } = require('http-proxy-middleware');
module.exports = function(app){
  app.use('/stream', createProxyMiddleware({ 
    target: 'http://localhost:5007', changedOrigin: true,
    })
  );
  app.use('/showMask', createProxyMiddleware({ 
    target: 'http://localhost:5007', changedOrigin: true,
    })
  );
  app.use('/userButton', createProxyMiddleware({ 
    target: 'http://localhost:5007', changedOrigin: true,
    })
  );
  app.use('/setMyEmotion', createProxyMiddleware({ 
    target: 'http://localhost:5007', changedOrigin: true,
    })
  );
  app.use('/myEmotion', createProxyMiddleware({ 
    target: 'http://localhost:5007', changedOrigin: true,
    })
  );
  app.use('/getServerData', createProxyMiddleware({ 
    target: 'http://localhost:5007', changedOrigin: true,
    })
  );
}