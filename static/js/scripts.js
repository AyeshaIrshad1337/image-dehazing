document.getElementById('process-button').addEventListener('click', function() {
    var formData = new FormData();
    var fileInput = document.getElementById('file');
    var modelSelect = document.getElementById('model-select');
    var loadingMessage = document.getElementById('loading-message');
    
    if (fileInput.files.length > 0) {
        formData.append('file', fileInput.files[0]);
        formData.append('model', modelSelect.value);

        // Show loading message
        loadingMessage.style.display = 'block';

        fetch('/dehaze/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': csrfToken
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.dehazed_image_url) {
                document.getElementById('dehazed-img').src = data.dehazed_image_url;
                document.getElementById('download-link').href = data.dehazed_image_url;
            }
            // Hide loading message
            loadingMessage.style.display = 'none';
        })
        .catch(error => {
            console.error('Error:', error);
            // Hide loading message
            loadingMessage.style.display = 'none';
        });
    } else {
        alert("Please upload an image first.");
    }
});

document.getElementById('file').addEventListener('change', function(event) {
    var file = event.target.files[0];
    if (file) {
        var uploadedFilename = document.getElementById('uploaded-filename');
        uploadedFilename.textContent = `Uploaded File: ${file.name}`;
    }
});
