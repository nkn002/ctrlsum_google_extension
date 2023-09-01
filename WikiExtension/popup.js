$(function() {
    $('#keywordsubmit').click(function() {
        var search_topic = $('#keyword').val();
        var website_content = ""; // Initialize website_content

        // Retrieve the URL of the active tab
        chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
            if (tabs && tabs.length > 0) {
                var currentTab = tabs[0];
                website_content = currentTab.url; // Assign the URL to website_content

                if (search_topic) {
                    chrome.runtime.sendMessage({
                        topic: search_topic,
                        content: website_content
                    }, function(response) {
                        var result = response.farewell;
                        alert(result.summary);

                        var notifOptions = {
                            type: "basic",
                            iconUrl: "icon128.png",
                            title: "Summary For Your Result",
                            message: result.summary
                        };

                        chrome.notifications.create('SumNotif', notifOptions);
                    });
                }
            }
        });

        $('#keyword').val('');
    });
});
