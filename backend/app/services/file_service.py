import aiofiles
import os

class FileService:
    async def save_file(self, file):
        save_dir = "uploads"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file.filename)
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        return {"filename": file.filename, "path": file_path}
