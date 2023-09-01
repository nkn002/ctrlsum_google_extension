var serverhost = 'http://127.0.0.1:8000';

    
	chrome.runtime.onMessage.addListener(
		function(request, sender, sendResponse) {
			var url = serverhost + '/wiki/get_wiki_summary/';
            console.log(request);
            fetch(url, {
                method: "POST",
                mode: "cors",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({topic:request.topic, content:request.content}),
            })
            .then(response => response.json())
			.then(response => sendResponse({farewell: response}))
			.catch(error => console.log(error))

//             return response.json();
				
			return true;  // Will respond asynchronously.
		  
	});

